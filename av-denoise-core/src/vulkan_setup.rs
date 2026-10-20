use crate::vulkan_features::ExtendedFeatures;
use ash::vk;
use cubecl::wgpu::{WgpuDevice, WgpuDeviceKind, WgpuSetup};

pub fn create(selector: &WgpuDevice) -> Result<(WgpuSetup, u32), anyhow::Error> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let (kind, index) = match selector.kind {
        WgpuDeviceKind::DiscreteGpu(index) => (Some(wgpu::DeviceType::DiscreteGpu), index),
        WgpuDeviceKind::IntegratedGpu(index) => (Some(wgpu::DeviceType::IntegratedGpu), index),
        WgpuDeviceKind::VirtualGpu(index) => (Some(wgpu::DeviceType::VirtualGpu), index),
        WgpuDeviceKind::Other(index) => (Some(wgpu::DeviceType::Other), index),
        WgpuDeviceKind::Cpu => (Some(wgpu::DeviceType::Cpu), 0),
        _ => (None, 0),
    };
    let mut adapters = cubecl::future::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    if kind.is_none() {
        adapters.sort_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        });
    }
    let adapter = adapters
        .into_iter()
        .filter(|adapter| kind.is_none_or(|kind| adapter.get_info().device_type == kind))
        .nth(index)
        .ok_or_else(|| anyhow::anyhow!("Vulkan adapter {selector:?} is unavailable"))?;
    let features = adapter
        .features()
        .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
    let mut limits = adapter.limits();
    let queue_count;
    let (device, queue) = unsafe {
        let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>().unwrap();
        let shared = hal.shared_instance();
        let mut extended = ExtendedFeatures::from_adapter(shared.raw_instance(), &hal, features);
        if !extended.has_required_features() {
            anyhow::bail!("Vulkan adapter lacks native CubeCL features");
        }
        let extensions = hal.required_device_extensions(features);
        let mut physical = hal.physical_device_features(&extensions, features);
        limits.max_immediate_size = hal
            .physical_device_capabilities()
            .properties()
            .limits
            .max_push_constants_size;
        if extended
            .index_64
            .is_some_and(|feature| feature.shader64_bit_indexing == 1)
        {
            limits.max_storage_buffer_binding_size = u64::MAX;
        }
        let core = shared
            .raw_instance()
            .get_physical_device_features(hal.raw_physical_device());
        let mut host_query = vk::PhysicalDeviceHostQueryResetFeatures::default();
        let mut supported = vk::PhysicalDeviceFeatures2::default().push_next(&mut host_query);
        shared
            .raw_instance()
            .get_physical_device_features2(hal.raw_physical_device(), &mut supported);
        if host_query.host_query_reset == 0 {
            anyhow::bail!("Vulkan adapter lacks host query reset for tonemapping");
        }
        queue_count = shared.raw_instance().get_physical_device_queue_family_properties(hal.raw_physical_device())[0].queue_count.min(2);
        let priorities = vec![1.0; queue_count as usize];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&priorities)];
        let names = extended
            .extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<Vec<_>>();
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&names);
        let info = physical
            .add_to_device_create(info)
            .enabled_features(&core)
            .push_next(&mut host_query);
        let info: tracel_ash::vk::DeviceCreateInfo<'_> = std::mem::transmute(info);
        let info = extended.add_to_device_create(info);
        let info: vk::DeviceCreateInfo<'_> = std::mem::transmute(info);
        let native = shared
            .raw_instance()
            .create_device(hal.raw_physical_device(), &info, None)?;
        let memory = wgpu::MemoryHints::MemoryUsage;
        let opened = hal.device_from_raw(native, None, &extensions, features, &limits, &memory, 0, queue_count - 1)?;
        let descriptor = wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            memory_hints: memory,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::enabled(),
        };
        adapter.create_device_from_hal(opened, &descriptor)?
    };
    Ok((WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: wgpu::Backend::Vulkan,
    }, queue_count))
}
