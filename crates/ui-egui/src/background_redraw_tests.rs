//! Background full CPU redraws (#4024). The GPU tests skip when there is no adapter.

use super::*;
use serde_json::json;

#[test]
fn blocks_cover_the_document_once() {
    let bounds = Rect::new(0, 0, 2500, 1100);
    let b = blocks(bounds);
    assert_eq!(b.len(), 3 * 2);
    assert_eq!(b.iter().map(|r| r.width() as u64 * r.height() as u64).sum::<u64>(), 2500 * 1100);
    for (i, r) in b.iter().enumerate() {
        assert!(r.x0 % BLOCK == 0 && r.y0 % BLOCK == 0, "block {i} starts on the grid: {r:?}");
        assert!(b[i + 1..].iter().all(|o| o.intersect(r).is_empty()), "block {i} overlaps another");
    }
}

#[test]
fn visible_blocks_come_first_nearest_the_view_centre() {
    let b = blocks(Rect::new(0, 0, 4096, 4096));
    // The view shows the right half of the second block row.
    let visible = Rect::new(2100, 1100, 4000, 1900);
    let o = order(&b, 0..b.len(), visible);
    assert_eq!(o.len(), 16);
    let shown: Vec<usize> = (0..b.len()).filter(|&i| !b[i].intersect(&visible).is_empty()).collect();
    assert_eq!(shown.len(), 2);
    assert!(o[..2].iter().all(|i| shown.contains(i)), "{o:?}");
    // Then the block row's neighbours, the far corners last.
    assert!(o[14..].contains(&12), "the bottom-left corner is among the last: {o:?}");
    // Only the blocks asked for.
    assert_eq!(order(&b, [3, 9].into_iter(), visible).len(), 2);
}

/// A kept composite of `doc`: `bytes` texels per block.
fn kept(doc: &Arc<Document>, bytes: usize) -> Kept {
    let all = blocks(doc.bounds());
    let texels: Vec<Vec<u8>> = all.iter().map(|_| vec![1u8; bytes]).collect();
    Kept {
        doc: Arc::downgrade(doc),
        id: doc.id,
        display: 7,
        format: eframe::wgpu::TextureFormat::Rgba8Unorm,
        bytes: (bytes * all.len()) as u64,
        blocks: all,
        texels,
    }
}

#[test]
fn kept_composites_come_back_only_for_the_exact_state_and_stay_bounded() {
    use eframe::wgpu::TextureFormat::{Rgba8Unorm, Rgba16Float};
    let mut a = Document::new("kept", photocraft_geom::Size::new(2000, 1500), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
    let mut layer = photocraft_doc::Layer::raster("one", a.pixel_format());
    if let photocraft_doc::LayerContent::Raster(s) = &mut layer.content {
        s.fill_rect(Rect::new(0, 0, 300, 300), &[0.2, 0.4, 0.6, 1.0]);
    }
    a.layers.push(layer);
    let a = Arc::new(a);
    let all = blocks(a.bounds());
    let changed = |f: &dyn Fn(&mut Document)| {
        let mut d = (*a).clone();
        f(&mut d);
        Arc::new(d)
    };
    let mut cache = KeptComposites::default();
    cache.store(kept(&a, 4), &a, u64::MAX);
    // The same state hits, also as an equal document that isn't the same Arc (a restore after a solo).
    assert!(cache.find(&a, 7, Rgba8Unorm, &all).is_some());
    assert!(cache.find(&Arc::new((*a).clone()), 7, Rgba8Unorm, &all).is_some());
    // Any change to what the composite depends on misses.
    let hidden = changed(&|d| d.layers[0].visible = false);
    let faded = changed(&|d| d.layers[0].opacity = 0.5);
    let multiplied = changed(&|d| d.layers[0].blend = photocraft_color::BlendMode::Multiply);
    let painted = changed(&|d| {
        if let photocraft_doc::LayerContent::Raster(s) = &mut d.layers[0].content {
            s.fill_rect(Rect::new(10, 10, 11, 11), &[1.0, 0.0, 0.0, 1.0]);
        }
    });
    for (what, d) in [("visibility", &hidden), ("opacity", &faded), ("blend mode", &multiplied), ("pixels", &painted)] {
        assert!(cache.find(d, 7, Rgba8Unorm, &all).is_none(), "{what} changed");
    }
    assert!(cache.find(&a, 8, Rgba8Unorm, &all).is_none(), "another display transform");
    assert!(cache.find(&a, 7, Rgba16Float, &all).is_none(), "another texture format");
    assert!(cache.find(&a, 7, Rgba8Unorm, &all[1..]).is_none(), "other blocks");
    // At most KEEP entries, the least recently used going first.
    cache.store(kept(&hidden, 4), &hidden, u64::MAX);
    assert!(cache.find(&a, 7, Rgba8Unorm, &all).is_some());
    cache.store(kept(&faded, 4), &faded, u64::MAX);
    assert_eq!(cache.entries.len(), KEEP);
    assert!(cache.find(&hidden, 7, Rgba8Unorm, &all).is_none(), "the least recently used went");
    assert!(cache.find(&a, 7, Rgba8Unorm, &all).is_some() && cache.find(&faded, 7, Rgba8Unorm, &all).is_some());
    // Storing a state again replaces its entry.
    cache.store(kept(&faded, 4), &faded, u64::MAX);
    assert_eq!(cache.entries.len(), 2);
    // The byte cap: a smaller one drops the oldest, and a composite over it isn't kept.
    let one = kept(&a, 4).bytes;
    cache.trim(one);
    assert_eq!(cache.entries.len(), 1);
    assert!(cache.bytes() <= one);
    cache.trim(0);
    cache.store(kept(&a, 4), &a, one - 1);
    assert!(cache.entries.is_empty(), "over the cap");
    // A closed document's composites go.
    cache.store(kept(&a, 4), &a, u64::MAX);
    cache.retain(|id| id != a.id);
    assert!(cache.entries.is_empty());
    // A state History no longer holds isn't kept alive: its entry is dead and goes.
    let gone = changed(&|d| d.layers[0].opacity = 0.25);
    cache.store(kept(&gone, 4), &gone, u64::MAX);
    let equal = Arc::new((*gone).clone());
    drop(gone);
    assert!(cache.find(&equal, 7, Rgba8Unorm, &all).is_none(), "no strong reference to the drawn state");
    cache.trim(u64::MAX);
    assert!(cache.entries.is_empty());
}

#[test]
fn the_reduced_picture_goes_to_the_mip_level_the_view_samples() {
    // The canvas shader samples level -log2(scale) when zoomed out (trilinear between the two
    // around it): the picture goes to the finer one.
    assert_eq!(preview_level(1.0), None);
    assert_eq!(preview_level(0.6), None);
    assert_eq!(preview_level(0.5), Some(1));
    assert_eq!(preview_level(0.3), Some(1));
    assert_eq!(preview_level(0.25), Some(2));
    assert_eq!(preview_level(0.1), Some(3));
    assert_eq!(preview_level(0.01), Some(MAX_PREVIEW_LEVEL));
    for bad in [0.0, -0.5, f32::NAN, f32::INFINITY] {
        assert_eq!(preview_level(bad), None, "{bad}");
    }
    assert_eq!(reduce(Rect::new(0, 1024, 2600, 1700), 4), Rect::new(0, 256, 650, 425));
    assert_eq!(reduce(Rect::new(1024, 0, 2047, 5), 2), Rect::new(512, 0, 1024, 3));
}

#[test]
fn the_reduced_picture_covers_only_blocks_still_to_draw_in_view() {
    let b = blocks(Rect::new(0, 0, 8192, 4096));
    let all: Vec<usize> = (0..b.len()).collect();
    // Zoomed out over the whole document, or onto part of it.
    assert_eq!(preview_area(&b, &all, Rect::new(-500, -500, 9000, 5000)), Rect::new(0, 0, 8192, 4096));
    assert_eq!(preview_area(&b, &all, Rect::new(1500, 900, 3000, 2100)), Rect::new(1024, 0, 3072, 3072));
    // Blocks already drawn are left out, and a view of none of the rest gets no picture.
    let rest: Vec<usize> = all.iter().copied().filter(|&i| b[i].x0 >= 4096).collect();
    assert_eq!(preview_area(&b, &rest, Rect::new(0, 0, 9000, 5000)), Rect::new(4096, 0, 8192, 4096));
    assert!(preview_area(&b, &rest, Rect::new(0, 0, 2000, 2000)).is_empty());
}

#[test]
fn the_reduced_document_scales_what_is_measured_in_document_pixels() {
    use photocraft_doc::vector::{Path, Subpath, VectorMask};
    use photocraft_doc::{Effect, LayerMask};
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 800, "height": 600})).expect("new");
    s.execute("layer.new.layer", json!({})).expect("layer");
    s.execute("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    s.execute("layer.layerStyle.dropShadow", json!({"distance": 12, "size": 16})).expect("shadow");
    let mut doc = (*s.active().expect("document").doc).clone();
    let l = doc.layers.last_mut().expect("layer");
    l.mask = Some(LayerMask { feather: 8.0, ..LayerMask::reveal_all() });
    l.vector_mask = Some(VectorMask { feather: 4.0, ..VectorMask::new(Path::new(vec![Subpath::polygon(&[(40.0, 40.0), (400.0, 40.0), (400.0, 320.0)])])) });
    // In a group too.
    let inner = l.clone();
    doc.layers.push(photocraft_doc::Layer::group("group", vec![inner]));
    let p = preview_document(&doc, 4);
    assert_eq!((p.size.width, p.size.height), (200, 150));
    for l in [&p.layers[1], &p.layers[2].children().expect("group")[0]] {
        match &l.effects.items[0] {
            Effect::DropShadow(sh) => assert_eq!((sh.distance, sh.size), (3.0, 4.0)),
            e => panic!("{e:?}"),
        }
        assert_eq!(l.mask.as_ref().map(|m| m.feather), Some(2.0));
        let v = l.vector_mask.as_ref().expect("vector mask");
        assert_eq!(v.feather, 1.0);
        let a = v.path.subpaths[0].knots[1].anchor;
        assert_eq!((a.x, a.y), (100.0, 10.0));
    }
    // The original is untouched.
    assert!(matches!(&doc.layers[1].effects.items[0], Effect::DropShadow(sh) if sh.size == 16.0));
}

#[test]
fn the_reduced_picture_of_a_stroked_styled_shape_matches_the_full_one() {
    // A stroked shape with an effect composites from its path (#5561 review): left at full
    // size, the reduced picture showed it four times too large.
    use photocraft_color::{Color, ColorMode, SampleType};
    use photocraft_doc::vector::{Path, ShapeLayer, ShapeStroke, Subpath};
    use photocraft_doc::{Effect, Fill};
    use photocraft_geom::Size;
    let mut doc = Document::with_background("shape", Size::new(256, 256), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    let mut sh = ShapeLayer {
        path: Path::new(vec![Subpath::polygon(&[(16.0, 16.0), (144.0, 16.0), (144.0, 144.0), (16.0, 144.0)])]),
        fill: Some(Fill::Solid(Color::BLACK)),
        stroke: Some(ShapeStroke { width: 8.0, paint: Fill::Solid(Color::rgb(1.0, 0.0, 0.0)), ..Default::default() }),
        // Feathered too: the blur is in document pixels.
        feather: 8.0,
        ..Default::default()
    };
    sh.cache = Some(photocraft_vector::render_shape(&sh, doc.pixel_format(), doc.bounds()));
    let mut layer = Layer::new("shape", LayerContent::Shape(sh));
    layer.effects.items = vec![Effect::default_drop_shadow()];
    doc.layers.push(layer);
    let full = photocraft_compose::render(&doc, doc.bounds());
    let p = preview_document(&doc, 4);
    let reduced = photocraft_compose::render(&p, p.bounds());
    // Every 4th pixel of the full picture, as the reduced one samples; only the anti-aliased
    // and blurred edges may differ.
    let differing = (0..64)
        .flat_map(|y| (0..64).map(move |x| (x, y)))
        .filter(|&(x, y)| reduced.px[y * 64 + x].iter().zip(&full.px[y * 4 * 256 + x * 4]).any(|(a, b)| (a - b).abs() > 0.1))
        .count();
    assert!(differing < 200, "{differing} of 4096 pixels differ");
    assert_eq!(reduced.px[48 * 64 + 48], full.px[192 * 256 + 192], "outside the shape");
    assert_eq!(reduced.px[20 * 64 + 20], full.px[80 * 256 + 80], "inside the shape");
}

#[test]
fn the_reduced_document_scales_the_pattern_tiles_of_effects() {
    // Glow pattern paint, bevel textures and pattern overlay phases are laid out in document
    // pixels too (#5561 review).
    use photocraft_doc::effects::{BevelTexture, FxCommon};
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 800, "height": 600})).expect("new");
    s.execute("layer.new.layer", json!({})).expect("layer");
    s.execute("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    s.execute("layer.layerStyle.outerGlow", json!({})).expect("glow");
    s.execute("layer.layerStyle.bevelEmboss", json!({})).expect("bevel");
    let mut doc = (*s.active().expect("document").doc).clone();
    let l = doc.layers.last_mut().expect("layer");
    for e in &mut l.effects.items {
        match e {
            Effect::OuterGlow(g) => g.paint = FxPaint::Pattern { name: "p".into(), id: String::new(), scale: 2.0 },
            Effect::BevelEmboss(b) => {
                b.texture = Some(BevelTexture { name: "p".into(), id: String::new(), scale: 2.0, depth: 1.0, invert: false, link: true, phase: (8.0, 4.0) })
            }
            _ => {}
        }
    }
    l.effects.items.push(Effect::PatternOverlay {
        common: FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
        name: "p".into(),
        id: String::new(),
        scale: 2.0,
        angle: 0.0,
        link: true,
        phase: (8.0, 4.0),
    });
    let p = preview_document(&doc, 4);
    let items = &p.layers.last().expect("layer").effects.items;
    assert_eq!(items.len(), 3);
    for e in items {
        match e {
            Effect::OuterGlow(g) => assert!(matches!(g.paint, FxPaint::Pattern { scale: 0.5, .. }), "{:?}", g.paint),
            Effect::BevelEmboss(b) => {
                let t = b.texture.as_ref().expect("texture");
                assert_eq!((t.scale, t.phase), (0.5, (2.0, 1.0)));
            }
            Effect::PatternOverlay { scale, phase, .. } => assert_eq!((*scale, *phase), (0.5, (2.0, 1.0))),
            e => panic!("{e:?}"),
        }
    }
}

#[test]
fn the_reduced_picture_of_a_pattern_fill_matches_the_full_one() {
    // A pattern fill renders from its full-size tile (#5561 review): left at full scale, the
    // reduced picture's tiles were four times too large.
    use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
    use photocraft_doc::pattern::Pattern;
    use photocraft_geom::Size;
    let mut doc = Document::with_background("pattern", Size::new(256, 256), ColorMode::Rgb, SampleType::U8, Color::WHITE);
    // 16 × 16 tile, red left half, blue right, drawn at half size from x = 12.
    let mut tile = photocraft_raster::Surface::new(PixelFormat::RGBA8);
    tile.fill_rect(Rect::new(0, 0, 8, 16), &[1.0, 0.0, 0.0, 1.0]);
    tile.fill_rect(Rect::new(8, 0, 16, 16), &[0.0, 0.0, 1.0, 1.0]);
    let pat = Pattern::new("rb", tile, 16, 16);
    let fill = Fill::Pattern { name: "rb".into(), scale: 2.0, id: pat.id.clone(), angle: 0.0, link: false, phase: (12.0, 0.0) };
    doc.patterns.push(pat);
    doc.layers.push(Layer::new("pattern", LayerContent::Fill(fill)));
    let full = photocraft_compose::render(&doc, doc.bounds());
    let p = preview_document(&doc, 4);
    let reduced = photocraft_compose::render(&p, p.bounds());
    for x in 0..64 {
        let (a, b) = (reduced.px[10 * 64 + x], full.px[(10 * 4 + 2) * 256 + x * 4 + 2]);
        assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() < 0.01), "x = {x}: {a:?} vs {b:?}");
    }
}

#[test]
fn a_cached_pattern_fill_keeps_its_cache_in_the_reduced_picture() {
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::FillCache;
    use photocraft_geom::Size;
    let mut doc = Document::new("cached", Size::new(64, 64), ColorMode::Rgb, SampleType::U8);
    let fill = Fill::Pattern { name: "gone".into(), scale: 1.0, id: String::new(), angle: 0.0, link: true, phase: (8.0, 4.0) };
    let mut l = Layer::new("pattern", LayerContent::Fill(fill.clone()));
    l.fill_cache = Some(FillCache { fill, surface: photocraft_raster::Surface::with_default(doc.pixel_format(), &[0.0, 1.0, 0.0, 1.0]) });
    doc.layers.push(l);
    let p = preview_document(&doc, 4);
    let l = &p.layers[0];
    let LayerContent::Fill(f) = &l.content else { panic!("{:?}", l.content) };
    assert!(matches!(f, Fill::Pattern { scale, phase, .. } if *scale == 0.25 && *phase == (2.0, 1.0)), "{f:?}");
    assert_eq!(l.fill_cache.as_ref().map(|c| &c.fill), Some(f), "the cache still matches");
    // A stale cache that the scaled fill would match is dropped, not drawn.
    let stale = Fill::Pattern { name: "gone".into(), scale: 0.25, id: String::new(), angle: 0.0, link: true, phase: (2.0, 1.0) };
    if let Some(c) = &mut doc.layers[0].fill_cache {
        c.fill = stale;
    }
    assert!(preview_document(&doc, 4).layers[0].fill_cache.is_none());
}

struct Gpu {
    harness: egui_kittest::Harness<'static, PhotocraftApp>,
    observer: GpuCanvas,
}

/// The app on a real device with background jobs, or `None` without a GPU adapter.
fn gpu_app() -> Option<Gpu> {
    let observer = Arc::new(std::sync::Mutex::new(None));
    let observer_out = observer.clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        egui_kittest::Harness::builder()
            .with_size(egui::vec2(1000.0, 700.0))
            .with_pixels_per_point(1.0)
            .with_max_steps(64)
            .wgpu_setup(crate::gpu_canvas::wgpu_setup())
            .build_eframe(move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                if let Some(rs) = cc.wgpu_render_state.as_ref() {
                    *observer_out.lock().expect("observer") = Some(GpuCanvas::new(rs));
                    app.set_wgpu(rs.clone());
                }
                app.background_jobs = true;
                app
            })
    }));
    let Ok(harness) = built else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    let observer = observer.lock().expect("observer").take()?;
    Some(Gpu { harness, observer })
}

/// A document the wgpu compositor leaves to the CPU (knockout on its top layer, which also has
/// Blend If).
fn cpu_only_document(app: &mut PhotocraftApp, w: u32, h: u32) -> DocId {
    app.run("file.new", json!({"width": w, "height": h, "background": "#336699"})).expect("new");
    app.run("layer.new.layer", json!({})).expect("layer");
    app.run("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    app.run("layer.layerStyle.blendingOptions", json!({"opacity": 60, "knockout": "shallow", "blendIf": {"channel": "gray", "underlying": [0, 10, 245, 255]}}))
        .expect("blending options");
    app.session.active().expect("document").doc.id
}

/// Frames until no redraw is running.
fn settle(g: &mut Gpu) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        g.harness.step();
        if g.harness.state().redraws.is_empty() {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "the redraw didn't finish");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// The canvas texture of `doc` against the synchronous CPU refresh of the current document.
fn assert_matches_synchronous(g: &mut Gpu, doc: DocId) {
    let app = g.harness.state_mut();
    let d = app.session.active().expect("document").doc.clone();
    let (display, _) = crate::canvas::canvas_display(app, &d, None);
    let reference = doc.0 ^ (1u64 << 60);
    g.observer.upload_composite(reference, &d, display.as_deref());
    let (_, size, shown) = g.observer.read_texels(doc.0).expect("canvas texture");
    let (_, _, want) = g.observer.read_texels(reference).expect("reference texture");
    assert_eq!(size, [d.size.width, d.size.height]);
    let differ = shown.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "{differ} texels differ from the synchronous refresh");
}

#[test]
fn a_full_cpu_refresh_draws_in_the_background_with_the_synchronous_pixels() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 2600, 1700);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "the frame started a redraw instead of compositing inline");
    settle(&mut g);
    let app = g.harness.state();
    assert_eq!(app.perf.last_refresh, "background");
    assert!(app.perf.gpu_fallback.as_deref().is_some_and(|f| f.contains("Advanced Blending")), "{:?}", app.perf.gpu_fallback);
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn an_edit_during_a_redraw_is_not_overwritten_by_blocks_drawn_before_it() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    // 4×3 blocks: the worker can't hand them all over before the next frame.
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc));
    // A corner, drawn last (farthest from the view's centre), changes on a layer of its own while
    // the redraw runs: a small edit, drawn at once.
    g.harness.state_mut().run("layer.new.layer", json!({})).expect("layer");
    g.harness.step();
    g.harness.state_mut().run("select.rect", json!({"x": 3900, "y": 2900, "width": 150, "height": 150})).expect("select");
    g.harness.step();
    g.harness.state_mut().run("edit.fill", json!({"color": "#22ee22"})).expect("fill");
    g.harness.step();
    assert_eq!(g.harness.state().perf.last_refresh, "rect", "the fill's damage was drawn synchronously");
    assert!(g.harness.state().redraws.contains_key(&doc), "the redraw goes on");
    settle(&mut g);
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn an_edit_damaging_most_of_a_large_document_is_drawn_in_the_background_too() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 2600, 1700);
    settle(&mut g);
    // Filling part of the Blend If layer can change the composite anywhere: the whole canvas.
    g.harness.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 300, "height": 300})).expect("select");
    g.harness.step();
    g.harness.state_mut().run("edit.fill", json!({"color": "#22ee22"})).expect("fill");
    assert_eq!(g.harness.state().session.active().expect("document").last_damage, Some(Rect::new(0, 0, 2600, 1700)));
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "not composited inline: {}", g.harness.state().perf.last_refresh);
    settle(&mut g);
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn showing_a_hidden_layer_again_reuses_the_redraw_from_before_it_was_hidden() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 2600, 1700);
    let layer = g.harness.state().session.active().and_then(|st| st.active_layer).expect("layer");
    settle(&mut g);
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    g.harness.state_mut().run("layer.setProps", json!({"layer": layer.0, "visible": false})).expect("hide");
    settle(&mut g);
    // Back to the state drawn before: its blocks come back, nothing is composited.
    g.harness.state_mut().run("layer.setProps", json!({"layer": layer.0, "visible": true})).expect("show");
    g.harness.step();
    assert!(g.harness.state().redraws.is_empty(), "no redraw: {}", g.harness.state().perf.last_refresh);
    assert_eq!(g.harness.state().perf.last_refresh, KEPT);
    assert_matches_synchronous(&mut g, doc);
    // A real change composites again. `last_refresh` is only set when that redraw finishes, so
    // check that one started, then that it was a composite rather than kept blocks.
    g.harness.state_mut().run("layer.setProps", json!({"layer": layer.0, "opacity": 0.3})).expect("opacity");
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "not served from kept blocks: {}", g.harness.state().perf.last_refresh);
    settle(&mut g);
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    assert_matches_synchronous(&mut g, doc);
}

/// Memory Usage lowered while a redraw runs: the finished composite is kept within what it leaves
/// now, not what it left when the redraw began.
#[test]
fn a_redraw_finishing_after_memory_usage_drops_keeps_nothing_over_it() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc));
    // Less than the document's own pixels: nothing is left for kept composites.
    g.harness.state_mut().session.edit_prefs(|p| p.performance.memory_usage_mb = 1);
    settle(&mut g);
    assert!(g.harness.state().kept.entries.is_empty(), "a composite was kept over the current cap");
}

#[test]
fn closing_the_document_stops_its_redraw() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc));
    g.harness.state_mut().run("file.close", json!({"save": false})).expect("close");
    g.harness.step();
    assert!(g.harness.state().redraws.is_empty());
}

#[test]
fn hiding_a_large_layer_over_the_gpu_budget_redraws_in_the_background() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    // A layer covering most, but not all, of the document: hiding it damages a large rect that
    // isn't the whole canvas.
    let app = g.harness.state_mut();
    app.run("file.new", json!({"width": 5000, "height": 2000, "background": "#336699"})).expect("new");
    app.run("layer.new.layer", json!({})).expect("layer");
    app.run("select.rect", json!({"x": 0, "y": 0, "width": 4800, "height": 1800})).expect("select");
    app.run("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    app.run("select.deselect", json!({})).expect("deselect");
    let (doc, layer) = {
        let st = app.session.active().expect("document");
        (st.doc.id, st.active_layer.expect("layer"))
    };
    settle(&mut g);
    assert!(g.harness.state().perf.last_refresh.starts_with("gpu"), "the document fits the GPU: {}", g.harness.state().perf.last_refresh);
    // A budget the pages of one 2048 px page cell fit (about 33 MB here) but not those of the
    // whole width (80 MB), as on a multi-gigabyte PSB. Damage refreshes keep it: `gpu_budget`
    // recomputes it for full ones only. A small edit still draws on the GPU at once.
    g.harness.state_mut().run("layer.new.layer", json!({})).expect("layer");
    g.harness.step();
    g.harness.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 64, "height": 64})).expect("select");
    g.harness.step();
    g.observer.set_memory_budget(48 << 20);
    g.harness.state_mut().run("edit.fill", json!({"color": "#22ee22"})).expect("fill");
    g.harness.step();
    assert_eq!(g.harness.state().perf.last_refresh, "gpu-rect", "{:?}", g.harness.state().perf.gpu_fallback);
    g.harness.state_mut().run("select.deselect", json!({})).expect("deselect");
    g.harness.step();
    assert!(g.harness.state().redraws.is_empty());
    g.observer.set_memory_budget(48 << 20);
    g.harness.state_mut().run("layer.setProps", json!({"layer": layer.0, "visible": false})).expect("hide");
    let damage = g.harness.state().session.active().expect("document").last_damage.expect("damage");
    assert!(damage != Rect::new(0, 0, 5000, 2000) && damage.width() as u64 * damage.height() as u64 >= crate::gpu_canvas::DEFER_PX, "{damage:?}");
    g.harness.step();
    assert!(g.harness.state().redraws.contains_key(&doc), "not composited inline: {}", g.harness.state().perf.last_refresh);
    settle(&mut g);
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    assert_eq!(g.harness.state().perf.gpu_fallback.as_deref(), Some(crate::gpu_canvas::OVER_BUDGET));
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn a_zoomed_out_redraw_shows_a_reduced_picture_before_its_blocks() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    // 4×3 blocks, fit in a 1000×700 window: about 22%, so level 2.
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    let level = g.harness.state().redraws.get(&doc).and_then(|r| r.preview).expect("a redraw with a reduced first picture");
    assert_eq!(level, 2);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while g.harness.state().redraws.get(&doc).is_some_and(|r| r.preview.is_some()) {
        assert!(std::time::Instant::now() < deadline, "the reduced picture didn't land");
        std::thread::sleep(std::time::Duration::from_millis(2));
        g.harness.step();
    }
    // It came first: most blocks are still to draw (the worker can only queue a few ahead).
    let left = g.harness.state().redraws.get(&doc).map_or(0, |r| r.left);
    assert!(left >= 6, "{left} blocks left when the reduced picture landed");
    let (_, size, early) = g.observer.read_level(doc.0, level).expect("level");
    assert_eq!(size, [1024, 768]);
    settle(&mut g);
    let (_, _, done) = g.observer.read_level(doc.0, level).expect("level");
    // The document is flat colour, so the reduced picture is already the finished level (the
    // texture starts out transparent).
    assert!(done.iter().all(|t| t[3] > 0.99), "the finished level is opaque");
    let differ = early.iter().zip(&done).filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| (x - y).abs() > 2.0 / 255.0)).count();
    assert_eq!(differ, 0, "{differ} texels of the reduced picture differ from the finished level");
    assert_eq!(g.harness.state().perf.last_refresh, "background");
    assert_matches_synchronous(&mut g, doc);
}

#[test]
fn a_finished_block_leaves_the_reduced_picture_of_its_pending_neighbours_alone() {
    pending_blocks_keep_the_reduced_picture(false);
}

#[test]
fn a_redraw_continued_on_the_same_state_keeps_putting_its_reduced_picture_back() {
    // Started again on the same document (#5561 review): the landed picture went with the old
    // redraw, so finished blocks' mip rebuilds left seams in their pending neighbours.
    pending_blocks_keep_the_reduced_picture(true);
}

/// Each block landing after the reduced picture leaves the picture whole under every pending
/// block; with `continued`, the redraw is first started again on the same state once it is in.
fn pending_blocks_keep_the_reduced_picture(continued: bool) {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut g) = gpu_app() else { return };
    // Flat colour: the reduced picture is opaque everywhere, and the texture starts transparent.
    let doc = cpu_only_document(g.harness.state_mut(), 4096, 3072);
    g.harness.step();
    let level = g.harness.state().redraws.get(&doc).and_then(|r| r.preview).expect("a redraw with a reduced first picture");
    let k = 1 << level;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let (mut checked, mut last_left, mut again, mut landed) = (0, usize::MAX, continued, false);
    while let Some(r) = g.harness.state().redraws.get(&doc) {
        landed |= r.shown.is_some();
        if again && landed && r.left > 1 {
            again = false;
            let ctx = g.harness.ctx.clone();
            let app = g.harness.state_mut();
            let d = app.session.active().expect("document").doc.clone();
            let (display, _) = crate::canvas::canvas_display(app, &d, None);
            start(app, &ctx, &g.observer, d.clone(), display, d.bounds(), 1.0, d.bounds(), None);
            assert!(app.redraws.get(&doc).is_some_and(|r| r.shown.is_some()), "the continued redraw keeps the landed picture");
            continue;
        }
        // After each of the first blocks to land once the picture is in: every pending block's
        // part of the level is still the opaque reduced picture, edges included.
        if landed && r.left < r.blocks.len() && r.left > 0 && r.left != last_left && checked < 4 {
            last_left = r.left;
            let pending: Vec<Rect> = r.blocks.iter().zip(&r.pending).filter(|(_, p)| **p).map(|(b, _)| reduce(*b, k)).collect();
            let (_, size, texels) = g.observer.read_level(doc.0, level).expect("level");
            for b in &pending {
                let clear = (b.y0..b.y1)
                    .flat_map(|y| (b.x0..b.x1).map(move |x| (x, y)))
                    .filter(|&(x, y)| texels[y as usize * size[0] as usize + x as usize][3] < 0.99)
                    .count();
                assert_eq!(clear, 0, "{clear} texels of pending block {b:?} (level {level}) lost the reduced picture");
            }
            checked += 1;
        }
        assert!(std::time::Instant::now() < deadline, "the redraw didn't finish");
        std::thread::sleep(std::time::Duration::from_millis(2));
        g.harness.step();
    }
    assert!(checked > 0, "no frame had both finished and pending blocks");
    assert!(!again, "the reduced picture never landed");
    assert_matches_synchronous(&mut g, doc);
}
