use smallvec::smallvec;
use std::sync::Arc;
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceExtensions, Queue,
    },
    image::SwapchainImage,
    instance::*,
    pipeline::GraphicsPipeline,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::ShaderModule,
    swapchain::*,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const VALIDATION_LAYERS: [&str; 1] = [
    "VK_LAYER_KHRONOS_validation", /* "VK_LAYER_LUNARG_api_dump", */
];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION: bool = false;

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
};

struct SwapchainData {
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

struct HelloTriangle {
    instance: Arc<Instance>,
    device: Arc<Device>,
    swapchain: SwapchainData,
    render_pass: Arc<RenderPass>,
    graphics_q: Arc<Queue>,
    surface_q: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
}

mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            vertex: {
                ty: "vertex",
                path: "shaders/vertex.glsl",
            },
            fragment: {
                ty: "fragment",
                path: "shaders/fragment.glsl",
            }
        }
    }
}

impl HelloTriangle {
    pub fn init() -> (Self, EventLoop<()>) {
        let instance = Self::create_instance();
        let (events_loop, surface) = Self::init_window(instance.clone());
        let physical = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_q, surface_q) = Self::create_device(physical, &surface);
        let (swapchain, images) = Self::create_swapchain(&device, &surface);
        let render_pass = Self::create_render_pass(device.clone(), &swapchain);
        let pipeline =
            Self::create_graphics_pipeline(device.clone(), swapchain.clone(), render_pass.clone());
        let framebuffers = Self::create_framebuffers(&swapchain, &render_pass, &images);

        (
            Self {
                instance,
                device,
                swapchain: SwapchainData {
                    swapchain,
                    images,
                    framebuffers,
                },
                render_pass,
                graphics_q,
                surface_q,
                pipeline,
            },
            events_loop,
        )
    }

    fn init_window(instance: Arc<Instance>) -> (EventLoop<()>, Arc<Surface<Window>>) {
        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(1280.0, 720.0))
            .build_vk_surface(&event_loop, instance)
            .unwrap();
        (event_loop, surface)
    }

    fn create_instance() -> Arc<Instance> {
        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions!");
        println!("Supported extensions: {:?}", supported_extensions);

        let required_extensions = vulkano_win::required_extensions();
        let mut create_info = InstanceCreateInfo::application_from_cargo_toml();
        create_info.enabled_extensions = required_extensions;

        if ENABLE_VALIDATION {
            create_info.enabled_layers =
                VALIDATION_LAYERS.iter().map(|s| (*s).to_owned()).collect();
        }

        Instance::new(create_info).expect("Failed to create Vulkan instance")
    }

    fn pick_physical_device<'a>(
        instance: &'a Arc<Instance>,
        surface: &Arc<Surface<Window>>,
    ) -> PhysicalDevice<'a> {
        use vulkano::device::Features;

        let is_device_suitable = |device: &PhysicalDevice| {
            let has_features = device.supported_features().is_superset_of(&Features {
                geometry_shader: true,
                ..Features::none()
            });

            let has_extensions = device
                .supported_extensions()
                .is_superset_of(&DEVICE_EXTENSIONS);

            let has_graphics_family = device
                .queue_families()
                .find(|&q| q.supports_graphics())
                .is_some();
            let has_present_family = device
                .queue_families()
                .find(|&q| q.supports_surface(surface).unwrap_or_default())
                .is_some();

            has_features && has_extensions && has_graphics_family && has_present_family
        };

        let device_score = |device: &PhysicalDevice| {
            let mut score = 0;
            if device.properties().device_type == PhysicalDeviceType::DiscreteGpu {
                score += 1000;
            }

            score += device.properties().max_image_dimension2_d;

            score
        };

        PhysicalDevice::enumerate(instance)
            .filter(is_device_suitable)
            .max_by_key(device_score)
            .expect("No suitable device found")
    }

    fn find_queue_families<'a>(
        physical: PhysicalDevice<'a>,
        surface: &Arc<Surface<Window>>,
    ) -> (QueueFamily<'a>, QueueFamily<'a>) {
        let graphics_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("No graphics queue family!");

        let surface_family = physical
            .queue_families()
            .find(|&q| q.supports_surface(surface).unwrap_or_default())
            .expect("No graphics queue family!");

        (graphics_family, surface_family)
    }

    fn create_device<'a>(
        physical: PhysicalDevice<'a>,
        surface: &Arc<Surface<Window>>,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        use vulkano::device::QueueCreateInfo;

        let (graphics_family, surface_family) = Self::find_queue_families(physical, surface);

        let create_infos = if surface_family == graphics_family {
            vec![QueueCreateInfo::family(graphics_family)]
        } else {
            vec![
                QueueCreateInfo::family(graphics_family),
                QueueCreateInfo::family(surface_family),
            ]
        };

        let (device, queues) = Device::new(
            physical,
            vulkano::device::DeviceCreateInfo {
                queue_create_infos: create_infos,
                enabled_extensions: physical.required_extensions().union(&DEVICE_EXTENSIONS),
                ..Default::default()
            },
        )
        .expect("Failed to create logical device");

        println!("Length: {}", queues.len());

        let queues: Vec<_> = queues.collect();

        let graphics_q = queues
            .iter()
            .find(|q| q.family() == graphics_family)
            .unwrap()
            .clone();
        let surface_q = queues
            .iter()
            .find(|q| q.family() == surface_family)
            .unwrap()
            .clone();

        (device, graphics_q, surface_q)
    }

    fn create_swapchain(
        device: &Arc<Device>,
        surface: &Arc<Surface<Window>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        use vulkano::{format::Format, image::ImageUsage, sync::Sharing};

        let physical = device.physical_device();

        let formats = physical
            .surface_formats(&surface, Default::default())
            .expect("No surface formats available!");
        let (image_format, color_space) = formats
            .iter()
            .find(|(image_format, color_space)| {
                image_format == &Format::B8G8R8A8_SRGB && color_space == &ColorSpace::SrgbNonLinear
            })
            .unwrap_or(formats.first().expect("No surface formats!"));

        let mut present_modes = physical
            .surface_present_modes(&surface)
            .expect("No surface present mode");
        let present_mode = present_modes
            .find(|mode| mode == &PresentMode::Mailbox)
            .unwrap_or(PresentMode::Fifo);

        let surface_capabilities = physical
            .surface_capabilities(&surface, Default::default())
            .expect("No surface capabilities!");
        let dimensions = surface.window().inner_size();

        let min_image_count = (surface_capabilities.min_image_count + 1).min(
            surface_capabilities
                .max_image_count
                .unwrap_or(u32::max_value()),
        );

        let (graphics_family, surface_family) = Self::find_queue_families(physical, surface);

        let image_sharing = if graphics_family == surface_family {
            Sharing::Exclusive
        } else {
            Sharing::Concurrent(smallvec![graphics_family.id(), surface_family.id()])
        };

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format: Some(*image_format),
                image_color_space: *color_space,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::color_attachment(),
                image_sharing,
                present_mode,
                ..Default::default()
            },
        )
        .unwrap();

        (swapchain, images)
    }

    fn create_render_pass(
        device: Arc<Device>,
        swapchain: &Arc<Swapchain<Window>>,
    ) -> Arc<RenderPass> {
        use vulkano::{
            image::ImageLayout,
            render_pass::{
                AttachmentDescription, AttachmentReference, LoadOp, RenderPassCreateInfo, StoreOp,
                SubpassDescription,
            },
        };

        let attachments = vec![AttachmentDescription {
            format: Some(swapchain.image_format()),
            samples: vulkano::image::SampleCount::Sample1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::PresentSrc,
            ..Default::default()
        }];

        let subpasses = vec![SubpassDescription {
            color_attachments: vec![Some(AttachmentReference {
                attachment: 0,
                layout: ImageLayout::ColorAttachmentOptimal,
                ..Default::default()
            })],
            ..Default::default()
        }];

        let render_pass_info = RenderPassCreateInfo {
            attachments,
            subpasses,
            ..Default::default()
        };

        RenderPass::new(device, render_pass_info).unwrap()
    }

    fn create_graphics_pipeline(
        device: Arc<Device>,
        swapchain: Arc<Swapchain<Window>>,
        render_pass: Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        use vulkano::{
            pipeline::{
                graphics::{
                    input_assembly::{InputAssemblyState, PrimitiveTopology},
                    vertex_input::VertexInputState,
                    viewport::{Viewport, ViewportState},
                },
                PartialStateMode, StateMode,
            },
            render_pass::Subpass,
        };

        let vertex_shader: Arc<ShaderModule> =
            shaders::load_vertex(device.clone()).expect("Failed to load vertex shader module");
        let fragment_shader: Arc<ShaderModule> =
            shaders::load_fragment(device.clone()).expect("Failed to load fragment shader module");

        let vertex_input_state = VertexInputState::new();

        let input_assembly_state = InputAssemblyState {
            topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleList),
            primitive_restart_enable: StateMode::Fixed(false),
        };

        let extents = swapchain.image_extent();

        let viewport = Viewport {
            origin: [0., 0.],
            dimensions: [extents[0] as f32, extents[1] as f32],
            depth_range: 0.0..1.0,
        };

        GraphicsPipeline::start()
            .vertex_input_state(vertex_input_state)
            .input_assembly_state(input_assembly_state)
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
            .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device)
            .unwrap()
    }

    fn create_framebuffers(
        swapchain: &Arc<Swapchain<Window>>,
        render_pass: &Arc<RenderPass>,
        images: &Vec<Arc<SwapchainImage<Window>>>,
    ) -> Vec<Arc<Framebuffer>> {
        use vulkano::image::view::ImageView;

        let extent = swapchain.image_extent();

        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        extent,
                        layers: 1,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect()
    }

    fn record_command_buffer(
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffer: Arc<Framebuffer>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        use vulkano::command_buffer::{
            AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                vec![[0.0, 0.0, 1.0, 1.0].into()],
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .draw(3, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    pub fn draw_frame(&mut self) {
        use vulkano::{
            swapchain, sync,
            sync::{FlushError, GpuFuture},
        };

        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    println!("Out of Date!");
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            println!("Suboptimal");
        }

        let command_buffer = Self::record_command_buffer(
            &self.device,
            &self.graphics_q,
            &self.pipeline,
            self.swapchain.framebuffers.get(image_i).unwrap().clone(),
        );

        let draw_and_present = acquire_future
            .then_execute(self.graphics_q.clone(), command_buffer.clone())
            .unwrap()
            .then_signal_semaphore()
            .then_swapchain_present(
                self.surface_q.clone(),
                self.swapchain.swapchain.clone(),
                image_i,
            );
        match draw_and_present.flush() {
            Err(FlushError::OutOfDate) => {
                println!("Out of date!");
                unsafe {
                    draw_and_present.signal_finished();
                }
                return;
            }
            Err(e) => {
                println!("{:?}", e);
            }
            Ok(()) => (),
        };

        match draw_and_present.then_signal_fence_and_flush() {
            Ok(future) => {
                future.wait(None).unwrap();
            }
            Err(FlushError::OutOfDate) => {
                println!("Fence - Out of date!");
            }
            Err(e) => {
                println!("Fence - Failed to flush future: {:?}", e);
            }
        }
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                println!("Resize");
            }
            Event::MainEventsCleared => {
                self.draw_frame();
            }
            _ => (),
        });
    }
}

fn main() {
    let (app, event_loop) = HelloTriangle::init();
    app.main_loop(event_loop);
}
