use smallvec::smallvec;
use std::{iter, sync::Arc};
use vulkano::{
    command_buffer::{
        pool::standard::*, synced::*, PrimaryAutoCommandBuffer, PrimaryCommandBuffer,
    },
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
    sync::*,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

// const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_LUNARG_api_dump"];

// const VALIDATION_LAYERS: [&str; 0] = [];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION: bool = false;

const FRAMES_IN_FLIGHT: usize = 2;

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
};

struct SwapchainData {
    swapchain: Arc<Swapchain<Window>>,
    _images: Vec<Arc<SwapchainImage<Window>>>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

struct FrameInFlight {
    command_buffer: Option<PrimaryAutoCommandBuffer>,
    fence: Option<Fence>,
    acquire_semaphore: Semaphore,
    draw_semaphore: Semaphore,
}

struct HelloTriangle {
    frames_in_flight: Vec<FrameInFlight>,
    current_frame: usize,

    _instance: Arc<Instance>,
    device: Arc<Device>,
    swapchain: SwapchainData,
    _render_pass: Arc<RenderPass>,
    graphics_q: Arc<Queue>,
    surface_q: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
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

        let frames_in_flight = iter::from_fn(|| {
            Some(FrameInFlight {
                command_buffer: None,
                fence: None,
                acquire_semaphore: Semaphore::from_pool(device.clone()).unwrap(),
                draw_semaphore: Semaphore::from_pool(device.clone()).unwrap(),
            })
        })
        .take(FRAMES_IN_FLIGHT)
        .collect();

        (
            Self {
                frames_in_flight,
                current_frame: 0,

                _instance: instance,
                device,
                swapchain: SwapchainData {
                    swapchain,
                    _images: images,
                    framebuffers,
                },
                _render_pass: render_pass,
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
    ) -> PrimaryAutoCommandBuffer {
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

        builder.build().unwrap()
    }

    // unsafe fn record_command_buffer(
    //     pipeline: &Arc<GraphicsPipeline>,
    //     framebuffer: Arc<Framebuffer>,
    //     pool: &Arc<StandardCommandPool>,
    // ) -> (StandardCommandPoolAlloc, Arc<SyncCommandBuffer>) {
    //     use vulkano::command_buffer::{
    //         pool::*, synced::*, sys::*, CommandBufferLevel, CommandBufferUsage, SubpassContents,
    //     };

    //     let pool_builder_alloc = pool
    //         .allocate(CommandBufferLevel::Primary, 1)
    //         .unwrap()
    //         .next()
    //         .unwrap();
    //     let mut builder = SyncCommandBufferBuilder::new(
    //         &pool_builder_alloc.inner(),
    //         CommandBufferBeginInfo {
    //             usage: CommandBufferUsage::OneTimeSubmit,
    //             ..Default::default()
    //         },
    //     )
    //     .unwrap();

    //     builder
    //         .begin_render_pass(
    //             RenderPassBeginInfo {
    //                 clear_values: vec![[0.0, 0.0, 1.0, 1.0].into()],
    //                 ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
    //             },
    //             SubpassContents::Inline,
    //         )
    //         .unwrap();
    //     builder.bind_pipeline_graphics(pipeline.clone());
    //     builder.draw(3, 1, 0, 0);
    //     builder.end_render_pass();

    //     (
    //         pool_builder_alloc.into_alloc(),
    //         Arc::new(builder.build().unwrap()),
    //     )
    // }

    pub fn draw_frame(&mut self) {
        use vulkano::{command_buffer::submit::*, swapchain, sync::*};

        let frame = &mut self.frames_in_flight[self.current_frame];

        if let Some(ref mut fence) = &mut frame.fence {
            fence.wait(None).unwrap();
            fence.reset().unwrap();
            frame.command_buffer.take();
        } else {
            frame.fence = Some(Fence::from_pool(self.device.clone()).unwrap());
        }

        unsafe {
            let acquired = match swapchain::acquire_next_image_raw(
                &*self.swapchain.swapchain,
                None,
                Some(&frame.acquire_semaphore),
                None,
            ) {
                Ok(acquired) => acquired,
                Err(AcquireError::OutOfDate) => {
                    println!("out of date!");
                    // no use waiting for this frame.
                    frame.fence.take();
                    return;
                }
                Err(e) => {
                    println!("Err: {:?}", e);
                    frame.fence.take();
                    return;
                }
            };

            if acquired.suboptimal {
                println!("Suboptimal");
            }

            let command_buffer = Self::record_command_buffer(
                &self.device.clone(),
                &self.graphics_q.clone(),
                &self.pipeline,
                self.swapchain
                    .framebuffers
                    .get(acquired.id)
                    .unwrap()
                    .clone(),
            );

            let mut queue_submit = SubmitCommandBufferBuilder::new();
            queue_submit.add_wait_semaphore(
                &frame.acquire_semaphore,
                PipelineStages {
                    fragment_shader: true,
                    ..PipelineStages::none()
                },
            );
            queue_submit.add_command_buffer(command_buffer.inner());
            queue_submit.add_signal_semaphore(&frame.draw_semaphore);
            queue_submit.set_fence_signal(frame.fence.as_ref().unwrap());
            queue_submit.submit(&*self.graphics_q).unwrap();

            let mut queue_present = SubmitPresentBuilder::new();
            queue_present.add_wait_semaphore(&frame.draw_semaphore);
            queue_present.add_swapchain(&*self.swapchain.swapchain, acquired.id as u32, None);
            match queue_present.submit(&*self.surface_q) {
                Ok(()) => (),
                Err(SubmitPresentError::OutOfDate) => {
                    println!("submit: out of date!");
                }
                Err(e) => {
                    println!("submit: {:?}", e);
                }
            }

            frame.command_buffer = Some(command_buffer);
            self.current_frame = (self.current_frame + 1) % self.frames_in_flight.len();
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

impl Drop for FrameInFlight {
    fn drop(&mut self) {
        if let Some(ref mut fence) = self.fence.as_mut() {
            fence.wait(None).unwrap();
            self.command_buffer.take();
        }
    }
}

fn main() {
    let (app, event_loop) = HelloTriangle::init();
    app.main_loop(event_loop);
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
