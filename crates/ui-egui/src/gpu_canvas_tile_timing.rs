//! Opt-in timing of the canvas texture tile size (#5767): what 8192 and 2048 px tiles cost on
//! this machine's GPU. Ignored, with no thresholds; it prints its measurements for a PR or an
//! issue. Run it on an optimized build, one test at a time:
//!
//! ```text
//! cargo test --release -p photocraft-ui-egui --lib tile_timing -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `PHOTOCRAFT_TILE_TIMING_SIZE=WxH` (default 16000x10000) and `PHOTOCRAFT_TILE_TIMING_RUNS`
//! (default 3) change the document and the number of runs. The two tile sizes alternate within a
//! run, each on a fresh app and device. `PHOTOCRAFT_GPU_TILE` must be unset: it overrides the
//! preference being measured.
//!
//! The two measurements stay separate:
//! - **GPU path**: a document the wgpu compositor draws (Clouds + 4% noise). Texture bytes and
//!   count; the first frame (composite + upload into fresh textures, rendered); then the median
//!   and worst CPU time of [`FRAMES`] UI frames (`step`: layout plus recording the canvas draw,
//!   one draw per tile) at Fit on Screen and at 100%. GPU execution time isn't measured: the
//!   harness's `render` reads the window back and costs about 0.5 s whatever is drawn.
//! - **CPU fallback**: a document the compositor leaves to the CPU (knockout + Blend If), redrawn
//!   in the background. Every UI frame until the redraw is done: total time, the worst frame and
//!   which one it was (frame 0 creates the textures; later ones upload blocks, and the first
//!   write into a fresh tile pays the driver's zero-fill), and frames over 50 ms.

use super::*;
use serde_json::json;
use std::time::{Duration, Instant};

const TILES: [u32; 2] = [8192, 2048];
const FRAMES: usize = 30;

fn size() -> [u32; 2] {
    std::env::var("PHOTOCRAFT_TILE_TIMING_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some([w.parse().ok()?, h.parse().ok()?])
        })
        .unwrap_or([16000, 10000])
}

fn runs() -> usize {
    std::env::var("PHOTOCRAFT_TILE_TIMING_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3)
}

/// The app on a real device with `tile` px canvas tiles, egui told the device's texture limit as
/// eframe does, or `None` without an adapter.
fn app(tile: u32) -> Option<(egui_kittest::Harness<'static, crate::PhotocraftApp>, GpuCanvas)> {
    let observer = std::sync::Arc::new(std::sync::Mutex::new(None));
    let observer_out = observer.clone();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        egui_kittest::Harness::builder()
            .with_size(egui::vec2(1000.0, 700.0))
            .with_pixels_per_point(1.0)
            .with_max_steps(64)
            .wgpu_setup(wgpu_setup())
            .build_eframe(move |cc| {
                crate::PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                app.session.edit_prefs(|p| p.performance.cache_tile_size = tile);
                if let Some(rs) = cc.wgpu_render_state.as_ref() {
                    *observer_out.lock().expect("observer") = Some(GpuCanvas::new(rs));
                    app.set_wgpu(rs.clone());
                }
                app.background_jobs = true;
                app
            })
    }));
    let Ok(mut harness) = built else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    let observer = observer.lock().expect("observer").take()?;
    harness.input_mut().max_texture_side = Some(observer.max_texture_side());
    Some((harness, observer))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn tile_count(size: [u32; 2], tile: u32) -> u32 {
    size[0].div_ceil(tile) * size[1].div_ceil(tile)
}

/// Median and worst CPU time of `FRAMES` UI frames, each rendered outside the timing.
fn steady(h: &mut egui_kittest::Harness<'static, crate::PhotocraftApp>) -> (f64, f64) {
    let mut t: Vec<f64> = (0..FRAMES)
        .map(|_| {
            let t0 = Instant::now();
            h.step();
            let t = ms(t0.elapsed());
            h.render().expect("render");
            t
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[FRAMES / 2], t[FRAMES - 1])
}

fn gpu_path(tile: u32, size: [u32; 2]) -> Option<String> {
    let (mut h, observer) = app(tile)?;
    let app = h.state_mut();
    app.run("file.new", json!({"width": size[0], "height": size[1]})).expect("new");
    app.run("filter.render.clouds", json!({})).expect("clouds");
    app.run("filter.noise.addNoise", json!({"amount": 4})).expect("noise");
    app.run("view.fitOnScreen", json!({})).expect("fit");
    let doc = app.session.active().expect("document").doc.id;
    let t0 = Instant::now();
    h.step();
    h.render().expect("render");
    let first = ms(t0.elapsed());
    let refresh = h.state().perf.last_refresh;
    assert!(refresh.starts_with("gpu"), "not on the GPU path: {refresh} ({:?})", h.state().perf.gpu_fallback);
    let (_, bytes) = observer.texture_info(doc.0).expect("canvas texture");
    let (fit, fit_max) = steady(&mut h);
    h.state_mut().run("view.actualPixels", json!({})).expect("100%");
    let (full, full_max) = steady(&mut h);
    Some(format!(
        "  gpu path     tile {tile:>4}: {:>3} textures, {bytes} bytes; first frame {first:.0} ms; UI frame at fit {fit:.1} ms median ({fit_max:.1} worst), at 100% {full:.1} ms ({full_max:.1})",
        tile_count(size, tile)
    ))
}

fn cpu_fallback(tile: u32, size: [u32; 2]) -> Option<String> {
    let (mut h, _observer) = app(tile)?;
    let app = h.state_mut();
    app.run("file.new", json!({"width": size[0], "height": size[1], "background": "#336699"})).expect("new");
    app.run("layer.new.layer", json!({})).expect("layer");
    app.run("edit.fill", json!({"color": "#cc8844"})).expect("fill");
    app.run("layer.layerStyle.blendingOptions", json!({"opacity": 60, "knockout": "shallow", "blendIf": {"channel": "gray", "underlying": [0, 10, 245, 255]}}))
        .expect("blending options");
    let doc = app.session.active().expect("document").doc.id;
    let (start, deadline) = (Instant::now(), Instant::now() + Duration::from_secs(300));
    let mut frames = Vec::new();
    loop {
        let t0 = Instant::now();
        h.step();
        frames.push(ms(t0.elapsed()));
        if frames.len() > 1 && !h.state().redraws.contains_key(&doc) {
            break;
        }
        assert!(Instant::now() < deadline, "the redraw didn't finish");
        std::thread::sleep(Duration::from_millis(2));
    }
    let total = ms(start.elapsed());
    // "background" at the measured size; a small document redraws inline ("full").
    let kind = h.state().perf.last_refresh;
    let (at, worst) = frames.iter().copied().enumerate().fold((0, 0.0), |a, (i, f)| if f > a.1 { (i, f) } else { a });
    let slow = frames.iter().filter(|&&f| f > 50.0).count();
    Some(format!(
        "  cpu fallback tile {tile:>4}: {kind} redraw {total:.0} ms over {} frames; worst frame {worst:.0} ms (frame {at}); {slow} frames > 50 ms",
        frames.len()
    ))
}

#[test]
#[ignore = "timing harness, run by hand (see the module docs)"]
fn tile_timing() {
    assert!(std::env::var_os("PHOTOCRAFT_GPU_TILE").is_none(), "unset PHOTOCRAFT_GPU_TILE: it overrides the tile size being measured");
    let _gpu = photocraft_testkit::gpu_lock();
    let (size, runs) = (size(), runs());
    let Some((_, observer)) = app(TILES[0]) else { return };
    let info = observer.adapter_info();
    eprintln!(
        "tile timing: {}x{} RGBA8, runs: {runs}, {} build; adapter {} ({:?}, {:?}, driver {} {}), max texture {}",
        size[0],
        size[1],
        if cfg!(debug_assertions) { "debug" } else { "release" },
        info.name,
        info.backend,
        info.device_type,
        info.driver,
        info.driver_info,
        observer.max_texture_side()
    );
    drop(observer);
    // The first frames on this machine compile shaders and warm the driver: measure neither.
    for tile in TILES {
        let _ = (gpu_path(tile, [1024, 1024]), cpu_fallback(tile, [1024, 1024]));
    }
    for run in 0..runs {
        eprintln!("run {}", run + 1);
        // Alternate which size goes first, so neither always gets the warmer machine.
        let order = if run % 2 == 0 { TILES } else { [TILES[1], TILES[0]] };
        for line in order.map(|tile| gpu_path(tile, size)).into_iter().chain(order.map(|tile| cpu_fallback(tile, size))).flatten() {
            eprintln!("{line}");
        }
    }
}
