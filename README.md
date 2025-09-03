# ✏️Doodle to Magic

📢 2025년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다


## 소개

본 프로젝트는 개인의 낙서 그림을 기반으로 3D 캐릭터를 자동 생성하는 것을 목표로 합니다. 

기존에는 낙서를 3D 실물로 제작하기 위해 많은 시간과 비용이 드는 수작업에 의존해야 했으며, 이 과정에서 낙서 고유의 창의성이 희석될 수 있었습니다. 이러한 문제를 해결하기 위해, 우리는 인공지능 기술을 활용하여 불완전한 낙서 형태를 해석해 완성도 높은 2D 캐릭터로 다듬고, 이를 기반으로 정확하고 매끄러운 3D 모델을 자동 생성하는 파이프라인을 구축하고자 합니다. 이 기술을 통해 누구나 자신의 상상력이 담긴 낙서를 손쉽게 디지털 창작물이나 실제 장난감으로 구현할 수 있게 될 것입니다.

구체적인 목표는 다음과 같습니다.
1) 낙서 고유의 특성을 유지한 아마추어 스타일 2D 이미지 변환
2) 낙서 고유의 특성을 유지한 포켓몬 스타일 2D 이미지 변환 후 3D 형태의 object로 변환
3) End-to-End 파이프라인 구성

## 방법론
**Pipeline**
<img width="950" height="257" alt="스크린샷 2025-09-02 오후 11 20 14" src="https://github.com/user-attachments/assets/5464af49-da8e-4a5c-a339-680531e692c0" />

 본 프로젝트는 Amateur Dataset과 Pokemon Dataset을 통해 finetuning한 control_v11p_sd15_scribble 모델을 통해 사용자가 직접 그린 Doodle을 Childlike Style 2D Image와 Pokemon Style 2D Image로 변환합니다. 이후, 3D Model인 TripoSR을 통해 Pokemon Style 2D Image를 입체적인 3D mesh로 변환합니다. 


**Data pre-processing**

<img width="868" height="356" alt="스크린샷 2025-09-03 오전 12 18 44" src="https://github.com/user-attachments/assets/bd5274cf-79e8-4d45-9e8d-1924db4f638b" />
<img width="908" height="331" alt="스크린샷 2025-09-02 오후 10 11 10" src="https://github.com/user-attachments/assets/b6d238f0-8658-4f58-b87f-84ea1ac48351" />

**Training(LoRA)**
<img width="927" height="296" alt="스크린샷 2025-09-02 오후 11 22 02" src="https://github.com/user-attachments/assets/7aa8d218-e7ae-4c59-b588-9550e648355a" />



**Finetuning Dataset**
| **데이터셋** | **설명** |
| --- | --- |
| [AMATEUR dataset](https://huggingface.co/datasets/keshan/amateur_drawings-controlnet-dataset) | 낙서 그림, 그리고 이를 segmentation한 그림, caption이 pair로 있는 데이터셋 |
| [Poketmon dataset](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) | poketmon 그림과 각 그림에 대한 caption이 달려있는 데이터 |



**Prior Research**
- control_v11p_sd15_scribble: https://huggingface.co/lllyasviel/sd-controlnet-scribble
- TripoSR: https://github.com/VAST-AI-Research/TripoSR

## 환경 설정

### Conda

### Directory
(Requirements, Anaconda, Docker 등 프로젝트를 사용하는데에 필요한 요구 사항을 나열해주세요)

## 사용 방법

(프로젝트 실행 방법 (명령어 등)을 적어주세요.)

## 예시 결과

1) 낙서 → 포켓몬 형태의 2D 이미지
<img width="800" height="560" alt="image" src="https://github.com/user-attachments/assets/9e418497-9ecf-4f38-ade7-5761fc0d3f6a" />


2) 포켓몬 2D 이미지 → 3D (오른쪽 면, 왼쪽 면)
<img width="715" height="430" alt="image" src="https://github.com/user-attachments/assets/ea51c4ea-09f3-4c01-bbad-7a2bd919dc36" />



## 팀원

- [신명경] https://github.com/w-shin1112
- [김윤서] https://github.com/hiyseo
- [김태관] https://github.com/TTKKWAN
- [백승현] https://github.com/snghyeon100
