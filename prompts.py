'''
Here is where the prompts live
'''

user_prompt = """
summarize the following text:
LLaVA-PruMerge:
Adaptive Token Reduction for Efficient Large
Multimodal Models
Yuzhang Shang1,2,∗
Mu Cai1,∗
Bingxin Xu2
Yong Jae Lee1,†
Yan Yan2,†
1University of Wisconsin–Madison
2Illinois Institute of Technology
https://llava-prumerge.github.io
Abstract
Large Multimodal Models (LMMs) have shown significant reasoning capabilities
by connecting a visual encoder and a large language model. LMMs typically
use a fixed amount of visual tokens, such as the penultimate layer features in
the CLIP visual encoder, as the prefix content. Recent LMMs incorporate more
complex visual inputs, such as high-resolution images and videos, which increase
the number of visual tokens significantly. However, due to the design of the
Transformer architecture, computational costs associated with these models tend to
increase quadratically with the number of input tokens. To tackle this problem, we
explore a token reduction mechanism and find, similar to prior work, that many
visual tokens are spatially redundant. Based on this, we propose PruMerge, a novel
adaptive visual token reduction approach, which largely reduces the number of
visual tokens while maintaining comparable model performance. We first select
the unpruned visual tokens based on their similarity to class tokens and spatial
tokens. We then cluster the pruned tokens based on key similarity and merge
the clustered tokens with the unpruned tokens to supplement their information.
Empirically, when applied to LLaVA-1.5 [Liu et al., 2023a], our approach can
compress the visual tokens by 14.4 times on average, and achieve comparable
performance across diverse visual question-answering and reasoning tasks. Code
and checkpoints will be released.
1
Introduction
Large Language Models (LLMs) [OpenAI, 2023b, Team et al., 2023, Jiang et al., 2023, Touvron
et al., 2023] have shown strong reasoning abilities. LLMs are usually high-capacity Transformers
pretrained with a large-scale text corpus. Large Multimodal Models (LMMs), inherit the LLMs for
text generation, but also leverage a visual encoder such as CLIP-ViT [Radford et al., 2021] to embed
image patches into visual tokens as the prefix visual context.
LMMs need substantial computation to conduct inference. The LLM is the primary factor for the high
computation cost, since the visual encoder is usually quite small relative to the LLM. For example,
the commonly used CLIP visual encoder, ViT-L, only has 0.3B parameters, while the corresponding
LLM such as LLaMA [Touvron et al., 2023] or Vicuna [Vicuna, 2023] can have 7B or 13B parameters.
As a result, reducing the LLM’s inference cost is the key to achieving low LMM inference cost.
Prior works [Chu et al., 2023, 2024, Yuan et al., 2023a] mainly focus on replacing the LLM backbone
with a smaller language model with less parameters, such as Phi-2 [Javaheripi et al., 2023]. However,
such approaches sacrifice the reasoning abilities of LLMs, leading to a large performance gap in
∗Equal Contribution. † Equal Advising Author. Work done during Yuzhang’s visit to UW-Madison.
arXiv:2403.15388v1  [cs.CV]  22 Mar 2024
Vision Encoder
Projector 𝑾
Token PruMerge
Tokenizer
Vision Input 𝑿𝒗
Language Response 𝒀𝒂
𝒁𝒗
𝒁𝒗′
𝑯𝒗
𝑯𝒒
Language Model 𝒇𝜽
Language Instruction
𝑿𝒒
(a) Main idea of LLaVA-PruMerge.
(b) PruMerged Token Visualization.
Figure 1: (a) We prune and merge the visual tokens coming from the vision encoder, while keeping
all other procedures of the LMM the same. By reducing the number of visual tokens, our proposed
method, PruMerge, significantly reduces the computation cost for text generation in LMMs. (b) The
visualizations of the attentive tokens. We design a token reduction method to adaptively select visual
tokens based on the information density of the visual input, enabling the LLM to perceive visual
input effectively and efficiently. More attentive tokens are sampled in complex images such as ones
with text, while fewer are sampled on simpler images. Besides, such attentive tokens are usually
located at the regions with dense information.
visual question-answering and reasoning such as VQAv2 and MM-Bench [Chu et al., 2024]. A
similar approach is to apply quantization for LLMs [Liu et al., 2023b, Yuan et al., 2024].
However, the cost of LLMs comes from not only its large number of parameters, but also the length
of the input context due to the quadratic complexity of the Transformer’s attention operation. The
context length in LMMs is especially important, where a fixed amount of visual tokens serves as
the prefixed tokens. For example, in LLaVA-1.5, 576 visual tokens are appended, leading to high
training and inference costs. Thus, an intriguing question is: Can we reduce the number of prefix
visual tokens while maintaining comparable performance?
In our study, we find that such visual tokens are redundant, similar to findings in previous related
work Bolya et al. [2023], Liu et al. [2022], and most of the visual tokens can be pruned without
largely sacrificing the performance. In particular, we find that the activations are very sparse upon
the similarity matrix between the class token and spatial patches, which indicates that only a small
amount of the visual tokens are related to the key visual information in the image. Motivated by this,
we use this similarity to select important visual tokens. Specifically, we leverage the Interquartile
Range (IQR) [Boukerche et al., 2020] scoring function in outlier detection to prune the visual tokens.
Moreover, we merge the visual tokens using k-nearest neighbor and update the sampled visual tokens
via weighted averaging, which further enhances performance. Finally, we optionally finetune the
LLM to let the model better adapt to our token deduction design.
Empirically, LLaVA-PruMerge can effectively and adaptively prune the visual tokens in each image
in LLaVA-1.5 [Liu et al., 2023a], where with just 6.9% of visual tokens, which is around 40 tokens on
average, our model can maintain comparable performance with that of retaining all 576 tokens across
diverse benchmarks. Our work demonstrates the effectiveness of building efficient large multimodal
models from the perspective of visual token pruning and paves the road for further research.
2
Related Work
2.1
Large Multimodal Models (LMMs)
Large Language Models (LLMs) such as GPT-4 [OpenAI, 2023b], LLaMA [Touvron et al., 2023],
Mistral [Jiang et al., 2023], and Gemini [Team et al., 2023] have demonstrated strong question
2
answering and reasoning capabilities over text. Large Multimodal Models (LMMs) [Liu et al., 2023b,
Zhu et al., 2023, Yin et al., 2023, Zhang et al., 2024] extend these reasoning capabilities to images,
where given an image and an associated question, a vision encoder and an LLM are leveraged to
generate text responses in a chat format. More recent works extend whole-image understanding
into region-level understanding [Cai et al., 2024, Zhang et al., 2023b, Peng et al., 2023, Chen et al.,
2023], video understanding [Lin et al., 2023, Zhang et al., 2023a] and 3D scene understanding [Hong
et al., 2023]. Such works typically feed the visual tokens directly into the LLM as prefix tokens, via
either MLP [Liu et al., 2023a], Qformer [Dai et al., 2023, Zhu et al., 2023], or resampler [Alayrac
et al., 2022]. The number of visual tokens can be prohibitively long, especially when the images
are high-resolution [Liu et al., 2024, OpenAI, 2023a]. In this paper, we reduce the number of visual
tokens by leveraging the similarity between the class token and the spatial patch tokens.
2.2
Efficient LMMs
The need for cross-modal capabilities in resource-limited scenarios has become increasingly impor-
tant. Despite advancements in LMMs, their large-scale training and deployment incur significant
computational costs, necessitating efficient parallel device implementations. Google’s Gemini [Team
et al., 2023] is a leader in efficient LMMs, achieving state-of-the-art performance on multimodal
benchmarks and introducing mobile-scale LMMs suitable for low-memory devices. However, Gemini
remains closed-source. Open-source initiatives, like LLaVA-1.5 [Liu et al., 2023a], utilize advanced
compression techniques, such as 4/8 bit quantization [Dettmers et al., 2022, Shang et al., 2024].
Further efforts towards efficient LMMs include MobileVLM [Chu et al., 2023], which develops a
compact LLM and an efficient multimodal feature projector, and its successor, MobileVLM-v2 [Chu
et al., 2024], which explores improved training strategies for mobile scenarios. TinyGPT-V [Yuan
et al., 2023a] leverages the advanced Phi-2 [Javaheripi et al., 2023] LLM to surpass the perfor-
mance of significantly larger models. Similarly, LLaVA-Phi [Zhu et al., 2024] and Vary-toy [Wei
et al., 2024] introduce smaller backbones and enhanced vocabularies for broader generalizability.
TinyLLaVA [Zhou et al., 2024] investigates the impacts of architectural choices, data quality, and
training strategies, demonstrating that smaller LMMs can match the performance of their larger
counterparts with optimized data and training. MoE-LLaVA [Lin et al., 2024] adapts Mixture of
Experts (MoE) to mitigate model degradation due to sparsity.
2.3
Token Reduction
The notorious squared complexity in Transformers [Vaswani et al., 2017] is a well-known prob-
lem, as it is one of the key bottlenecks in scaling the sequence length. Sparse attention such as
Linformer [Wang et al., 2020] and ReFormer [Kitaev et al., 2020] reduce the quadratic attention
complexity by conducting attention operations within a certain region rather than the full context.
Token merging [Bolya et al., 2023] utilizes full attention but gradually reduces the number of tokens
in each transformer block by selecting the most representative tokens with bipartite matching. In
recent LMMs [Liu et al., 2023b, Zhu et al., 2023], prefix visual tokens serve as a fixed budget for
context, which becomes one of the leading factors for their efficiency. In our study, we find that
by simply pruning and merging visual tokens based on their similarity, we can achieve comparable
performance using less than one tenth of the original tokens.
3
Method: Token Pru-Merging
In this section, we first review the basic implementation of large mutilmodal models (LMMs), with
a particular focus on the visual encoder component (i.e., Vision Transformer). We highlight the
direct correlation between the number of visual tokens and the efficiency of LMMs (Sec. 3.1). Next,
we present a plug-and-play token reduction method specifically designed for LMMs, called token
PruMerge. Our method features two key components: (1) Adaptive Important Token Selection (AITS)
via Outlier Detection which adaptively determines the optimal number of visual tokens to retain
based on the unique characteristics of the image (Sec. 3.2); and (2) Token Supplement (TS) via
Similar Key Clustering, which facilitates more efficient processing without compromising the model’s
performance by maintaining the integrity and richness of the visual information (Sec. 3.3).
...
"""

ai_response = """
Imagine a technology capable of understanding and following objects in videos, regardless of how lengthy the footage might be. This is the realm of MAVOS, a groundbreaking approach in the field of video object segmentation. Traditional methods face challenges with long videos, struggling under the weight of high memory demands and slow processing times. MAVOS offers a solution through an innovative modulated cross-attention memory mechanism, optimizing how visual information is processed and significantly enhancing efficiency.
This technology focuses on the essentials, ignoring extraneous data to maintain fast processing speeds and reduce memory usage by an impressive 87%, all while preserving accuracy. Such advancements in video analysis have far-reaching implications, from improving video editing tools to enhancing surveillance systems, and even advancing autonomous vehicle technologies. MAVOS represents a significant stride towards smarter, more efficient video understanding, showcasing the potential of integrating sophisticated memory management techniques in real-time applications.
"""