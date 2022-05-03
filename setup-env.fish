set VULKAN_SDK (dirname (status --current-filename))/x86_64
fish_add_path --path $VULKAN_SDK/bin
set -gx LD_LIBRARY_PATH $VULKAN_SDK/lib
set -gx VK_ADD_LAYER_PATH $VULKAN_SDK/etc/vulkan/explicit_layer.d


