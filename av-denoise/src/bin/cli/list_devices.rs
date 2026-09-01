use av_denoise::Device;
use av_denoise::accelerate::Accelerator;
use av_denoise::enumerate::{BackendDevices, enumerate_devices};

/// Lists the devices every backend in `enable` can see.
///
/// One row per device, naming the backends that offer it, so the
/// selector printed in the first column can be passed straight to
/// `--device`.
pub fn run_list_devices(enable: &[Accelerator]) -> String {
    format_table(&enumerate_devices(enable))
}

/// Renders what the backends reported as a table.
///
/// Kept apart from the enumeration so it can be tested without a GPU.
///
/// The table opens with a blank line. Graphics drivers write their own
/// notices to stderr while the backends start, and the header is hard to
/// pick out of that without a gap in front of it.
fn format_table(reported: &[BackendDevices]) -> String {
    let rows = build_rows(reported);
    let unavailable: Vec<&BackendDevices> = reported.iter().filter(|b| !b.available).collect();

    let mut out = String::from("\n");

    if rows.is_empty() {
        out.push_str("No usable devices found.\n");
    } else {
        let width = rows
            .iter()
            .map(|(device, _)| device.len())
            .chain(std::iter::once("DEVICE".len()))
            .max()
            .unwrap_or(0);

        out.push_str(&format!("{:<width$}  BACKENDS\n", "DEVICE"));
        for (device, backends) in &rows {
            out.push_str(&format!("{device:<width$}  {}\n", backends.join(", ")));
        }
    }

    if !unavailable.is_empty() {
        let names: Vec<String> = unavailable.iter().map(|b| b.accelerator.to_string()).collect();
        out.push_str(&format!(
            "\nEnabled but unavailable on this machine: {}\n",
            names.join(", "),
        ));
    }

    out
}

/// Turns per-backend device lists into per-device backend lists.
///
/// Rows are sorted by device kind and then by ordinal, and the backends
/// within a row keep the order they were asked in. Every backend that
/// started gets a `default` row, because `--device default` always
/// works on a backend that runs at all.
///
/// The ordinals are per-backend, so two backends sharing a row do not
/// necessarily share a physical card.
fn build_rows(reported: &[BackendDevices]) -> Vec<(String, Vec<String>)> {
    let mut rows: Vec<(Device, Vec<String>)> = Vec::new();

    let mut record = |device: Device, accelerator: Accelerator| {
        let name = accelerator.to_string();
        match rows.iter_mut().find(|(seen, _)| *seen == device) {
            Some((_, backends)) if backends.contains(&name) => {},
            Some((_, backends)) => backends.push(name),
            None => rows.push((device, vec![name])),
        }
    };

    for backend in reported.iter().filter(|b| b.available) {
        record(Device::Default, backend.accelerator);
        for device in &backend.devices {
            record(device.clone(), backend.accelerator);
        }
    }

    rows.sort_by_key(|(device, _)| sort_key(device));
    rows.into_iter()
        .map(|(device, backends)| (device.to_string(), backends))
        .collect()
}

/// Orders devices by kind first and ordinal second.
///
/// `default` leads because it is what a run without `--device` uses.
fn sort_key(device: &Device) -> (u8, usize) {
    match device {
        Device::Default => (0, 0),
        Device::Discrete { index } => (1, *index),
        Device::Integrated { index } => (2, *index),
        Device::Virtual { index } => (3, *index),
        Device::Cpu => (4, 0),
    }
}
