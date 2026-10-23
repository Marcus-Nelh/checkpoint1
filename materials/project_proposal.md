::: titlepage
**Project Proposal**\
**A Mixture of Experts Approach on Face Liveliness Detection for Face
Anti-Spoofing**\
**Marcus Nelhams**\

CS 228 - Biomtric Security with AI\
San Jose State University\
September 24, 2026
:::

# Review of Related Research

## Paper 1: On Analysis of Face Liveness Detection Mechanisms via Deep Learning Models

**Authors:** Syed Zoofa Rufai, Arvind Selwal, Deepika Sharma //
**Publisher:** IEEE

**Citation:** S. Z. Rufai, A. Selwal and D. Sharma, \"On Analysis of
Face Liveness Detection Mechanisms via Deep Learning Models,\" 2022
International Conference on Sustainable Computing and Data Communication
Systems (ICSCDS), Erode, India, 2022, pp. 59-64, doi:
10.1109/ICSCDS53736.2022.9760922.

### Summary

This survey paper goes over modern day techniques for face liveliness
detection including hardware approaches such as eye blinking and lip
moving detection and data driven machine learning methods with a focus
on deep learning models that are becoming more popular. The paper lays
out generalization as the most difficult problem to solve as
environment, camera, and mode of attack can all change in drastic ways
that make classifying across multiple datasets hard. It also mentions
the cost of deep learning models, both in creation and deployment,
leading to increased research in light weight models. The lack of
abundant training data is also mentioned as a substantial challenge when
trying to create a model that can generalize to real world use.

## Paper 2: Deep Learning Based Face liveness detection with a Lightweight CNN and depth detection using MiDaS

**Authors:** Niladri Das; Madhurendra Pratap Singh; Priyanka Biswas;
Nirmalya Kar; Swanirbhar Majumder // **Publisher:** IEEE

**Citation:** N. Das, M. P. Singh, P. Biswas, N. Kar and S. Majumder,
\"Deep Learning Based Face liveness detection with a Lightweight CNN and
depth detection using MiDaS,\" 2026 International Conference on
Intelligent Processing, Hardware, Electronics, and Radio Systems
(CIPHER), Jalandhar, India, 2026, pp. 1-6, doi:
10.1109/CIPHER70417.2026.11523927.

### Summary and Research Niche

This paper proposes a lightweight face liveness detection system that
uses a combination of a convolution neural network and a pretrained
depth estimation model called MiDaS to distinguish real faces from spoof
attacks. The CNN model is trained to classify an input as real or fake
based on facial features and structure while the MiDaS model tries to
determine a depth map of the face to better identify three dimensional
presentations from flat spoof attacks like printed photos or video
replay attacks. The final classification is decided by combining the
outputs of both models. The authors argue that the added depth analysis
addresses a major weakness in most modern liveliness detection systems
that rely solely on flat image inputs. The main strength of this paper
is that it identifies a way to resolve a major issue that is common in
most face liveliness detection systems that take flat images as inputs.
The main weakness of this paper is that the results were not compared
against a model that does not implement depth detection, and therefore
it is inconclusive if this model is better than a regular CNN model.

## Paper 3: Generalized Passive Face Liveness Detection Using Region-Specific Mixture-of-Experts

**Authors:** Takua Mokhamed, Manar Abu Talib, Qassim Nasir, Sohail Abbas
// **Publisher:** IEEE

**Citation:** T. Mokhamed, M. A. Talib, Q. Nasir and S. Abbas,
\"Generalized Passive Face Liveness Detection Using Region-Specific
Mixture-of-Experts,\" 2026 International Conference on Intelligent
Systems: Theories and Applications (SITA), Rabat, Morocco, 2025, pp.
1-6,

### Summary and Research Niche

This paper proposes a Mixture-of-Experts approach to face liveliness
detection, a strategy that utilizes multiple deep learning methods that
are each focused on a specific aspect or region of an input. This paper
argues that single model approaches struggle to generalize the vast
amount of facial cues and features that identify face liveliness. Each
\"expert\" aims to focus on one feature and/or attack vector, such as an
eye liveliness expert that tracks blinking and gaze changes that would
help circumvent static photo attacks. The results of all the experts are
then put through a gating network that then determines the final output.
The authors only implemented the skin-liveliness expert, but this expert
did show high accuracy when tested against an outside dataset. The main
strength of this paper is the demonstrated ability of the
skin-liveliness expert to accurately identify liveliness when tested
against an outside dataset. The main weakness is the fact that other
experts and the gating network were not implemented and therefore it
cannot be concluded that this MoE model would actually generalize better
than other models.

# Proposed Project

## Project Idea and Motivation

**Problem Statement:**

Face recognition systems can experience multiple different attack
vectors, including photo, video, mask, or deepfake based attacks. Face
liveliness detection attempts to determine if a face being presented to
the camera is a real human face. Different attack vectors have different
strengths and weaknesses in terms of the visual characteristics they try
to emulate. A mask based attack, for example, might inaccurately
represent the texture of a persons face, but will accurately represent
shadows and lighting. A video based attack, on the other hand, would
present a real face in video form, and therefore would not have any
errors in terms of facial features, but might cause artifacts in the
environment like incorrect shadows or glare. This project will
investigate weather combining the outputs of two models, one an
\"expert\" on facial texture/features and another an \"expert\" on
environmental factors, can provide a better system for detecting if a
face is an actual person or an attack.\
\
**Project Idea:**

Develop a mixture of experts approach for face liveliness detection.
This system will have two separate models, one focused on facial
features and textures and another focused on background and
environmental information. The outputs of these networks will then be
used to determine a final classification decision. The first paper
listed, On Analysis of Face Liveness Detection Mechanisms via Deep
Learning Models by Takua Mokhamed, et al., explains the overall field of
face liveliness detection as it stands, pointing out that neural
networks have become the best way to determine face liveliness over
other classification methods. This relates to this project as the two
models that will be developed will be neural networks, more
specifically, convolution neural networks. The second paper, Deep
Learning Based Face liveness detection with a Lightweight CNN and depth
detection using MiDaS by Niladri Das, et al., demonstrates a lightweight
and successful system that combines an anti-spoof model and a pretrained
depth prediction model to determine face liveliness. This paper
demonstrates that a multi-modal system, like the one laid out in this
proposal, can be implemented with success. The final paper Generalized
Passive Face Liveness Detection Using Region-Specific Mixture-of-Experts
by Takua Mokhamed, et al., lays out how a mixture of experts modal can
be constructed.\

## Key Modules and Components

1.  **Preprocessing:**

    The images from the data set will be preprocessed and normalized
    before being inputted into the models. Different preprocessing steps
    will be used to try and emphasize what each expert should be
    learning. For example, the face texture expert will be preprocessed
    in a way that emphasizes facial features and textures. The
    environmental expert model, on the other hand, will be processed in
    a way that emphasizes environmental characteristics like shadows,
    lighting and background.

2.  **Face \"Expert\":**

    The face expert will be a convolution neural network (CNN) trained
    to distinguish facial features and textures that can correctly
    identify a real presentation from an attack. Its input will be a
    preprocessed version of the input images that emphasizes facial
    features and textures and its output will be its decision on if the
    input is an attack or a real person.

3.  **Environment \"Expert\":**

    The environment expert will be a convolution neural network (CNN)
    that will aim to learn environmental features and cues that signify
    whether a presentation is genuine or an attack. The model will be
    trained to determine not only if a presentation is an attack or not,
    but also environmental conditions such as lighting and background to
    encourage it to learn environmental features that might be useful
    for liveliness detection.

4.  **Decision Fusion Module:**

    This will be a simple neural network that takes the outputs of the
    two experts models as inputs. It will then make a final
    classification, either genuine or spoof, based on the information
    provided from the two expert models.

## Tentative Project Timeline

[]{#tab:timeline label="tab:timeline"}

::: {#tab:timeline}
  **Timeframe**        **Key Activities**
  -------------------- -------------------------------------------------------------------------------
  Weak 7               Design/Outline Preprocessing Steps, Models, and Pipeline
  Weak 8 to Weak 9     Preprocess and Clean Data Set for Models
  Weak 10              Implement Initial Pipeline, Models, and Evaluation System
  Weak 11 to Weak 14   Iterate and Experiment with Different Parameters and Perform Initial Analysis
  Weak 15 to Weak 16   Implement Final Improvements, Write Report, and Create Presentation
:::

## Datasets and Data

**Primary Dataset:** CelebA-Spoof

**Source:** https://github.com/ZhangYuanhan-AI/CelebA-Spoof

**Description:**

CelebA-Spoof is a face anti-spoofing dataset that contains over 600,000
face images with a wide range of spoof conditions and types. Each image
contains annotated attributes like attack vector such as face-mask,
poster, tablet display, phone display, and photo. Moreover, each photo
has annotations on environment conditions such as lighting and
background as well as facial features such as eye color, hair color,
glasses, and more.
