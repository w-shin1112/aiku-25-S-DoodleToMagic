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
inputs 폴더 안에 input_1.png 형식으로 이미지 파일 준비 후 아래 script 실행
./run_test.sh 1 "cute tiger pokemon character"

혹은 배포된 gradio 사용

## 예시 결과

1) From Scribble to Amateur & Pokemon Style 2D Image
base model : lllyasviel/control_v11p_sd15_scribble
amateur prompt : a childlike crayon drawing, cute {input} character, no background
pokemon prompt : pokemon style, cute {input} pokemon character, no background

<img width="885" height="643" alt="image" src="https://github.com/user-attachments/assets/e25571dc-ccd4-407f-992c-50363f748622" />
<img width="950" height="658" alt="image" src="https://github.com/user-attachments/assets/8f280f81-38f1-4931-8b7a-a343e608a263" />
<img width="920" height="639" alt="image" src="https://github.com/user-attachments/assets/4255086b-c404-41a7-b0f4-912bae7cdcb5" />
<img width="915" height="648" alt="image" src="https://github.com/user-attachments/assets/aee2bbff-55a2-4435-84bf-b3f2411c85f9" />


2) From Pokemon 2D image to 3D (left and right)
<img width="895" height="549" alt="image" src="https://github.com/user-attachments/assets/2909d5f5-61e5-4efb-b6d2-162a2ae0a170" />

## Contribution 
1. Amatuer Style 2D Image task 에서 Baseline에 비해 원본 낙서 고유의 형태 유지
2. Pokemon Style 2D Image task 에서 Baseline에 비해 Pokemon 고유의 특성 재현
3. Pokemon 3D task 에서 얼굴 앞 뒤가 똑같아 보이지 않는 문제 해결 및 obj file을 viewr에 업로드 했을 때 색감이 흐릿한 문제 해결
<img width="871" height="292" alt="image" src="https://github.com/user-attachments/assets/ac33e928-f411-405a-92ed-04f9eda4e727" />

## Limitation 
1. Amateur Dataset prompt 및 segment 오류
2. Titan Gpu로 인한 제한적인 3D model
3. Pip library 호환성 문제로 인한 end-to-end pipleline 불가
4. 세심한 Texture 구현 불가
5. 정량적 평가 지표의 부재





## 팀원

- [신명경] https://github.com/w-shin1112
- [김윤서] https://github.com/hiyseo
- [김태관] https://github.com/TTKKWAN
- [백승현] https://github.com/snghyeon100
