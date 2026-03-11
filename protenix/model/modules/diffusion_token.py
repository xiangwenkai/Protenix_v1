# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Token-Level Diffusion Module

简化版的扩散模块，直接在token坐标上进行预测和去噪。
不需要AtomAttentionEncoder/Decoder，因为我们只预测token中心原子。
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LinearNoBias, Transition
from protenix.model.modules.transformer import DiffusionTransformer
from protenix.model.triangular.layers import LayerNorm


class TokenDiffusionConditioning(nn.Module):
    """
    Token-Level版本的DiffusionConditioning

    与原版类似，但输出和处理的都是token级别的表示
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        super(TokenDiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs

        # Pair conditioning (与原版相同)
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32
        )
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Single conditioning (与原版相同)
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )

        # Noise embedding (与原版相同)
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )

        # Single transitions (与原版相同)
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

    def prepare_cache(
        self,
        relp_feature: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """准备pair embedding缓存"""
        pair_z = torch.cat(
            tensors=[z_trunk, self.relpe(relp_feature)],
            dim=-1,
        )  # [..., N_token, N_token, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level: [..., N_sample]
            relp_feature: relative position encoding features
            s_inputs: [..., N_token, c_s_inputs]
            s_trunk: [..., N_token, c_s]
            z_trunk: [..., N_token, N_token, c_z]
            pair_z: cached pair embedding (可选)
            inplace_safe: 是否使用inplace操作
            use_conditioning: 是否使用conditioning

        Returns:
            single_s: [..., N_sample, N_token, c_s]
            pair_z: [..., N_token, N_token, c_z]
        """
        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        else:
            if inplace_safe:
                pair_z = pair_z.clone()

        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_token, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))

        # Noise embedding
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(single_s.dtype)  # [..., N_sample, c_noise_embedding]

        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(dim=-2)  # [..., N_sample, N_token, c_s]

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        return single_s, pair_z


class TokenDiffusionModule(nn.Module):
    """
    Token-Level Diffusion Module

    简化版扩散模块，直接在token坐标上操作：
    1. 输入: token坐标 [N_sample, N_token, 3]
    2. 处理: 使用token embedding和pair embedding
    3. 输出: 去噪后的token坐标 [N_sample, N_token, 3]

    移除了AtomAttention组件，因为不需要处理残基内部原子
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        transformer: dict[str, int] = {"n_blocks": 24, "n_heads": 16, "drop_path_rate": 0},
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(TokenDiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        # Conditioning module
        self.diffusion_conditioning = TokenDiffusionConditioning(
            sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        )

        # Token坐标编码: 将3D坐标编码为token特征
        self.coord_to_token = nn.Sequential(
            LinearNoBias(in_features=3, out_features=c_token, precision=torch.float32),
            LayerNorm(c_token, create_offset=False),
        )

        # Single embedding到token空间的投影
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros",
        )

        # Diffusion Transformer
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        # 输出层: 从token特征解码回坐标
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.token_to_coord = LinearNoBias(
            in_features=c_token,
            out_features=3,
            precision=torch.float32,
            initializer="zeros",
        )

        print(f"TokenDiffusionModule initialized with sigma_data={self.sigma_data}")

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """
        Token-level的前向传播

        Args:
            r_noisy: 归一化的token坐标 [..., N_sample, N_token, 3]
            t_hat_noise_level: [..., N_sample]
            input_feature_dict: 输入特征
            s_inputs: [..., N_token, c_s_inputs]
            s_trunk: [..., N_token, c_s]
            z_trunk: [..., N_token, N_token, c_z]
            pair_z: [..., N_token, N_token, c_z] (缓存)
            inplace_safe: 是否使用inplace操作
            chunk_size: chunk大小
            use_conditioning: 是否使用conditioning
            enable_efficient_fusion: 是否启用融合优化

        Returns:
            r_update: [..., N_sample, N_token, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        # Conditioning
        s_single, z_pair = self.diffusion_conditioning(
            t_hat_noise_level,
            input_feature_dict["relp"],
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            inplace_safe=inplace_safe,
            use_conditioning=use_conditioning,
        )  # s_single: [..., N_sample, N_token, c_s], z_pair: [..., N_token, N_token, c_z]

        # 将token坐标编码为特征
        a_token = self.coord_to_token(r_noisy)  # [..., N_sample, N_token, c_token]
        a_token = a_token.to(dtype=torch.float32)

        # 添加single embedding信息
        if inplace_safe:
            a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
        else:
            a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))

        # Transformer处理
        if enable_efficient_fusion:
            from protenix.model.utils import permute_final_dims
            z = LayerNorm(self.c_z, create_offset=False, create_scale=False)(z_pair.to(dtype=torch.float32))
            z = permute_final_dims(z, [2, 0, 1]).contiguous()
        else:
            z = z_pair.to(dtype=torch.float32)

        a_token = self.diffusion_transformer(
            a=a_token.to(dtype=torch.float32),
            s=s_single.to(dtype=torch.float32),
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        # 解码回坐标
        a_token = self.layernorm_a(a_token)
        r_update = self.token_to_coord(a_token)  # [..., N_sample, N_token, 3]

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """
        一步去噪: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy: [..., N_sample, N_token, 3]
            t_hat_noise_level: [..., N_sample]
            其他参数同f_forward

        Returns:
            x_denoised: [..., N_sample, N_token, 3]
        """
        # 归一化输入坐标 (EDM scaling)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        # 计算更新
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        # 重新缩放并组合 (EDM formula)
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None]
            / torch.sqrt(1 + s_ratio**2)
            * r_update
        ).to(r_update.dtype)

        return x_denoised
