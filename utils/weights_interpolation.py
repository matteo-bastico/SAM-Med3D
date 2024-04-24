import torch
import os.path as osp
import torch.nn.functional as F

from segment_anything.build_sam3D import sam_model_registry3D


sam_model = sam_model_registry3D["vit_b_mlp"](checkpoint=None)
print(sam_model)

src_path = "../ckpt/sam_med3d_turbo.pth"
out_path = osp.join(osp.dirname(src_path), "sam_vit_b_mlp.pth")

image_size = (256, 128, 512)
patch_size = 16
window_size = 14
encoder_depth = 12
encoder_global_attn_indexes = [2, 5, 8, 11]

# convert keys
token_len = (image_size[0] // patch_size, image_size[1] // patch_size, image_size[2] // patch_size)
window_len = window_size * 2 - 1

state_dict = torch.load(src_path, map_location=torch.device('cpu'))
for param in state_dict['model_state_dict'].keys():
    print(param)

model_state_dict = state_dict['model_state_dict']
print("Pos embedding shape before: ", model_state_dict["image_encoder.pos_embed"].shape)
old_weight = model_state_dict["image_encoder.pos_embed"]
model_state_dict["image_encoder.pos_embed"] = F.interpolate(old_weight.permute(0, 4, 1, 2, 3), size=token_len,
                                                      mode='trilinear').permute(0, 2, 3, 4, 1)
print("Pos embedding shape after: ", model_state_dict["image_encoder.pos_embed"].shape)

# h,w -> h,w,d
for vit_layer_idx in range(encoder_depth):
    key_d = "image_encoder.blocks.{}.attn.rel_pos_d".format(int(vit_layer_idx))
    key_h = "image_encoder.blocks.{}.attn.rel_pos_h".format(int(vit_layer_idx))
    key_w = "image_encoder.blocks.{}.attn.rel_pos_w".format(int(vit_layer_idx))
    target_size = (2 * token_len[0] - 1,
                   2 * token_len[1] - 1,
                   2 * token_len[2] - 1) if (vit_layer_idx in encoder_global_attn_indexes) else \
        (window_len, window_len, window_len)

    key_map = {
        key_d: key_d,
        key_h: key_h,
        key_w: key_w,
        # key_d: key_h,
    }  # target : src

    for dim, (tgt_key, src_key) in enumerate(key_map.items()):
        old_weight = model_state_dict[src_key]
        new_weight = F.interpolate(old_weight[None].permute(0, 2, 1), size=target_size[dim], mode='linear').permute(0, 2, 1)[
            0]
        print(src_key, "({})".format(old_weight.shape), "->", tgt_key, "({})".format(new_weight.shape))
        model_state_dict[tgt_key] = new_weight

torch.save(state_dict, out_path)
print("End")

'''
# convert all 256 -> 384

def reshape_tensor_with_copy_padding(tensor, target_shape_dim=128, new_dim_size=256):

#def reshape_tensor_with_copy_padding(tensor, target_shape_dim=256, new_dim_size=128):
    """
    Reshape a tensor's dimensions from target_shape_dim to new_dim_size,
    padding by copying slices from the tensor if the new size is larger.

    :param tensor: The input torch.Tensor to reshape.
    :param target_shape_dim: The dimension size to look for and change.
    :param new_dim_size: The new dimension size to reshape to.
    :return: A new tensor with the specified dimension(s) resized and padded by copying existing data if necessary.
    """
    shape = list(tensor.shape)

    # Find the index of the dimension that matches the target_shape_dim
    target_dim_idx = shape.index(target_shape_dim) if target_shape_dim in shape else -1

    # If the target dimension is not found or no padding is needed, return the tensor as is
    if target_dim_idx == -1 or shape[target_dim_idx] == new_dim_size:
        return tensor

    # Calculate how many elements we need to add to reach the new size
    padding_needed = new_dim_size - shape[target_dim_idx]
    # Calculate how many full copies can be made from the padding needed
    num_full_copies = padding_needed // shape[target_dim_idx]
    # Calculate the size of the last partial copy if needed
    partial_copy_size = padding_needed % shape[target_dim_idx]

    # Create a list of tensors to concatenate (starting with the original tensor)
    tensors_to_concat = [tensor]

    # Add full copies to the list
    for _ in range(num_full_copies):
        tensors_to_concat.append(tensor)

    # Add the last partial copy if needed
    if partial_copy_size > 0:
        partial_copy = tensor.narrow(target_dim_idx, 0, partial_copy_size)
        tensors_to_concat.append(partial_copy)

    # Concatenate all tensors along the target dimension
    reshaped_tensor = torch.cat(tensors_to_concat, dim=target_dim_idx)

    return reshaped_tensor


key_list = list(state_dict.keys())
for key in key_list:
    if (key.startswith("mask_decoder") or key.startswith("prompt")):
        del state_dict[key]

torch.save({"model_state_dict": state_dict}, out_path)
print("End")
'''





