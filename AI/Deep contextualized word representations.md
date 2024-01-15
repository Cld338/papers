---
tags:
  - NLP
Created: 2024-01-10
Creation Date: 2024-01-10
Last Modified: 2024-01-15
DOI: "\rhttps://doi.org/10.48550/arXiv.1802.05365"
완료 여부: false
---
## Abstract
&emsp;단어의 복잡한 특성(e.g, 문법 의미)와 이들이 언어적 맥락에 따라 어떻게 사용되는 지(i.e. 다의어)를 모두 모델링하는 새로운 유형의 깊은 문맥화된 단어 표현을 소개합니다. 우리의 단어 벡터는 대규모 말뭉치를 학습시킨 bidirectional Language Model(이하 biLM)의 내부 상태에 대해 학습된 함수입니다. 우리는 이러한 표현 방식이 다른 기존의 모델들에 도입될 수 있으며 question answering, textual entailment, sentiment analysis를 포함하는 6가지 까다로운 NLP 과제에서 최신 기술들을 현저하게 개선할 수 있음을 보입니다. 또한 우리는 사전 학습 모델의 내부 레이어를 노출시키는 것이 다운스트림 모델이 다양한 유형의 준감독 신호를 혼합할 수 있도록 하는 데 중요하다는 분석을 제시합니다.
## 1. Introduction
&emsp;사전 학습된 단어 표현은 많은 자연어를 이해하는 모델에서 핵심적인 요소입니다. 어쨌든, 고품질의 표현을 학습하는 것은 어려울 수 있습니다. 그들은 이상적으로 단어의 복잡한 특성(e.g., 문법과 의미)과 이들이 언어적 맥락에 따라 어떻게 사용되는지(i.e., 다의어를 모델링하는 것)를 모두 모델링할 수 있어야 합니다. 이 논문에서, 우리는 두 문제를 직접적으로 해결하고, 기존 모델에 쉽게 합쳐질 수 있고, 제시된 모든 어려운 언어 이해 문제들에서 최신 기술의 상당한 개선을 이룬 새로운 형태의 deep  contextualized word representation을 소개합니다.

&emsp;우리의 representation은 각 토큰에 전체 입력 문장에 대한 함수 표현이 할당된다는 점에서 기존 단어 유형 임베딩과 차이가 있습니다. <font color="#ff0000">우리는 대규모 말뭉치를 목적으로 학습된 짝지어진 언어 모델과 함께 학습된 양방향 LSTM으로 부터 얻은 벡터를 사용하였습니다</font>. 이러한 이유로, 우리는 그들을 ELMo(Embeddings from Language Models) representations라고 부릅니다. 문맥화된 단어 벡터를 학습시키는 이전의 방법들과 달리, ELMo representations는 양방향 LSTM의 모든 내부 레이어들의 함수라는 점에서 깊습니다. 더 구체적으로, 우리는 각 입력 단어 위에 쌓인 벡터의 선형 조합을 학습시켰고, 이는 LSTM의 최상위 레이어만을 사용할 때 보다 성능을 현저하게 개선시켰습니다.

 &emsp;이러한 방법으로 내부 상태들을 합치는 것은 매우 풍부한 단어 표현이 가능하도록 합니다. 고유한 평가를 사용하여, 우리는 상위 LSTM의 상태가 문맥의 의존성 측면을 포착하는 반면 하위 LSTM의 상태는 문법적 측면을 포착하는 것을 보였습니다. 동시에 모든 이러한 신호들을 노출시키는 것은 학습된 모델이 각 최종 작업에 가장 유용한 준감독 유형을 선택할 수 있으므로 매우 유익합니다.

 &emsp;광범위한 실험들을 통해 ELMo representation이 실제로 매우 잘 작동한다는 사실이 입증되었습니다. 우리는 먼저 그들이 question answering, textual entailment, sentiment analysis를 포함하는 6개의 다양하고 어려운 언어 이해 문제를 해결하기 위한 기존의 모델에 쉽게 추가될 수 있음을 보입니다. ELMo representation을 추가하는 것 만으로도 모든 케이스에서 최신 기술들을 상당히 개선시켰으며 상대 오류를 20% 이상 감소시켰습니다. 직접적인 비교가 가능한 과제에서, ELMo는 신경망 기계 번역 인코더를 사용해 계산하는 문맥화된 representation인 CoVe보다 높은 성능을 보였습니다. 마지막으로 ELMo와 CoVe에 대한 분석은 deep representation이 LSTM의 최상위 레이어로 부터 얻은 representation 보다 높은 성능을 가진다는 것을 보였습니다. 우리의 학습된 모델과 코드는 공개되었으며, 우리는 ELMo가 많은 다른 NLP 문제들에 대해 유사한 이점을 제공할 것이라고 예상합니다.
## 2. Related work
&emsp;라벨링 되지 않은 대규모 텍스트에서 문법적, 의미적 정보를 추출해내는 능력 덕분에, 사전 학습된 단어 벡터는 question answering, textual entailment, semantic role labeling 등을 포함하는 대부분의 최신 NLP 아키텍처에서 일반적인 구성 요소가 되었습니다. 어쨌든, 단어 벡터 학습을 위한 이러한 접근들은 각 단어마다 문맥에 비의존적인 하나의 representation만을 사용합니다.

&emsp;이전에 제시된 방법들은 하위 단어 정보를 풍부하게 하거나 각 단어의 의미에 대해 별도의 벡터를 학습하여 기존 단어 벡터가 가지는 몇몇 문제들을 극복하였습니다. 우리의 접근 방법은 문자 컨볼루션을 통해 부분 문자에 대한 이점을 가지고, 사전 정의된 의미를 예측하기 위해 명시적인 훈련 없이 다의어 정보를 원활하게 다운스트림으로 통합합니다.

&emsp;또한 다른 최근의 연구는 구문에 의존적인 representation을 학습시키는 것을 목표로 합니다. context2vec은 대상 주위의 문맥을 인코딩하기 위해 양방향 LSTM을 사용합니다. contextual embedding을 학습시키기 위한 다른 접근 방법들은 representation에 pivot word 자체를 포함하고 지도 학습 신경망 기계 번역(CoVe) 또는 비지도 학습 언어 모델의 인코더로 계산됩니다. 기계 번역 방식이 병렬 말뭉치의 크기에 제한을 받지만 이러한 방식들은 대규모 데이터셋에서 이점을 가집니다. 이 논문에서, 우리는 풍부한 단일 언어 데이터에 접근함으로써 얻는 최대한의 이점을 얻고, 약 3천만 개의 문장이 포함된 말뭉치로 biLM을 학습시킵니다. 또한 우리는 deep contextual representation에 대한 이러한 접근 방법들을 넓은 범위의 다양한 NLP 과제에 대해 잘 작동하는 것을 보여 일반화 시켰습니다.

&emsp;이전의 연구들은 양방향 RNN의 레이어들이 다른 종류의 정보를 인코딩하는 것을 보였습니다. 예를 들어, deep LSTM의 하위 레이어에 multi-task 구문 감독(e.g., 품사 태깅)을 도입하는 것은 종속성 구문 분석이나 CCG super tagging과 같은 높은 레벨의 task에서 전체적인 성능을 개선시킬 수 있습니다. RNN 기반 인코더-디코더 기계 번역 시스템에서, Belinkov et al. (2017)은 2-layer LSTM 인코더의 첫 번째 레이어에서 학습된 representation이 두 번째 레이어에 비해 품사 태그를 예측하는 데에 더 낫다는 것을 보였습니다. 마지막으로, 단어 문맥 인코딩을 위한 LSTM의 최상위 레이어는 단어의 의미에 대한 representation을 학습하는 것이 증명되었습니다. 우리는 ELMo representations의 수정된 언어 모델 목적에 의해 유사한 신호들이 유도됨을 보였고, 이는 다양한 유형의 준감독을 혼합하는 다운스트림 작업을 한 모델들을 학습하는 것에 매우 유익할 수 있습니다.

&emsp;Dai and Le (2015)와 Ramachandran et al. (2017)은 인코더-디코더 쌍을 언어 모델과 sequence autoencoder를 사용해 사전에 학습시켰고 task별 지도를 통해 fine tuning하였다. 이와 반대로, 우리는 biLM을 라벨링되지 않은 데이터로 학습시킨 뒤 가중치를 수정하고 task에 따라 모델 용량을 추가하여 다운스트림 훈련 데이터가 더 작은 지도 학습 모델을 지시하는 경우 크고 풍부하고 보편적인 biLM representation을 활용할 수 있었습니다.
## 3. ELMo: Embeddings from Language Models
&emsp;가장 널리 사용되는 단어 임베딩들과 달리, ELMo word representation은 이 섹션에 설명된 대로 전체 입력 문장에 대한 함수입니다 . 그들은 문자 컨볼루션을 통해 (Sec. 3.1), 내부 네트워크 상태에 대한 선형 함수로써 2-layer biLM의 최상위 layer에서 계산됩니다 (Sec. 3.2). 이 설정을 통해 우리는 biLM이 큰 규모에서 사전에 학습되어 있을 때 (Sec. 3.4) 준지도 학습을 사용할 수 있고, 넓은 범위의 기존의 neural NLP 아키텍처에 통합시킬 수 있습니다 (Sec. 3.3).

### 3.1 Bidirectional langualge models
&emsp; N개의 token $(t_1, t_2, \dots, t_N)$이 주어지면, 정방향 언어 모델은 $(t_1, \dots, t_{k-1})$이 주어졌을 때 $t_k$의 확률을 모델링하여 시퀀스의 확률을 계산합니다.
$$p(t_1, t_2, \dots, t_N)=\prod_{k=1}^{N} p(t_{k} | t_{1}, t_{2}, \dots t_{k-1})$$
최근의 최신 신경망 언어 모델은 토큰 임베딩 또는 문자 CNN에 대한 통해 문맥에 비의존적인 token representation $\mathrm{x}_{k}^{LM}$을 계산하고 정방향 LSTM의 $L$개의 layer를 통해 전달합니다. 각 위치 $k$에서, 각각의 LSTM layer는 $j=1,\dots,L$에서 문맥에 의존적인 representation $\overrightarrow{h}_{k,j}^{LM}$을 출력합니다. LSTM의 최상위 레이어의 출력은 Softmax layer를 통해 다음 token $t_{k+1}$을 예측하기 위해 사용됩니다.

&emsp;역방향 언어 모델은 이후의 문맥이 주어질 때 이전의 token을 예측하기 위해 반대 방향으로 문장을 받아들이는 것을 제외하면 정방향 언어 모델과 유사합니다.
$$p(t_1, t_2, \dots, t_N)=\prod_{k=1}^{N} p(t_{k} | t_{k+1}, t_{k+2}, \dots t_{N})$$
$(t_{k+1}, t_{k+2}, \dots t_{N})$이 주어질 때 $t_{k}$의 representation $\overleftarrow{h}_{k,j}^{LM}$를 만들어내는 $L$ layer backward LSTM의 레이어 $j$에서 정방향 언어 모델과 유사하게 구현될 수 있습니다.

&emsp;biLM은 정방향과 역방향 언어 모델을 결합합니다. 우리의 공식은 정방향과 역방향의 log likelihood를 모두 최대화합니다.
$$\sum_{k=1}^N(\log p(t_{k} | t_{1}, t_{2}, \dots t_{k-1};\Theta_{x}, \overrightarrow{\Theta}_{LSTM}, \Theta_{s}) + \\
\log p(t_{k} | t_{k+1}, t_{k+2}, \dots t_{N};\Theta_{x}, \overleftarrow{\Theta}_{LSTM}, \Theta_{s}))$$
우리는 각 방향의 LSTM이 가지는 parameter를 유지하면서 token representation($\Theta_{x}$)과 Softmax layer($\Theta_{s}$)에 필요한 parameter를 정방향과 역방향에 연결하였습니다. 완전히 독립된 parameter를 사용하는 대신에 일부 가중치를 공유한 다는 점을 제외하면 전체적으로 이 공식은 Peters et al. (2017)의 접근 방법과 유사합니다. 다음 섹션에서,  biLM의 선형 조합인 word representation을 학습하기 위한 새로운 방법을 도입하며 이전의 연구들로 부터 시작합니다.
## 3.2 ELMo
&emsp;ELMo는 task에 특이적인 biLM의 intermediate layer representation의 조합입니다. 각각의 token $t_{k}$에 대하여, $L$-layer biLM은 $2L+1$개의 representation을 계산합니다.
$$\begin{align}
R_{k} &= \{x_{k}^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} |j=1,\dots,L\}\\
&= \{h_{k,j}^{LM}|j=0, \dots, L\}
\end{align}$$
이때 $h_{k,0}^{LM}$은 token layer이고 각 양방향 LSTM에서 $h_{k,j}^{LM}=\left[\overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} \right]$입니다.

&emsp;다운스트림 모델에 도입하기 위해, ELMo는 $R$의 모든 레이어를 단일 벡터로 통합합니다 $\mathrm{ELMo}_{k}=E(R_{k};\Theta_{e})$. 가장 단순한 경우, ELMo는 TagLM (Peters et al., 2017) 그리고 CoVe (Mc-Cann et al., 2017) 와 같이 최상위 레이어를 선택합니다 $E(R_{k})=h_{k,L}^{LM}$. 보다 일반적으로, 우리는 task에 특이적인 모든 biLM 레이어의 가중치를 계산하였습니다.
$$\mathrm{ELMo}_{k}^{task}=E(R_{k};\Theta^task)=\gamma^{task} \sum_{j=0}^{L} s_{j}^{task} h_{k,j}^{LM} 
\qquad (1)$$
(1)에서,  $s^{task}$는 Softmax에 의해 정규화된 가중치이고 scalar parameter $\gamma^{task}$는 task model이 ELMo vector 전체를 수정할 수 있도록 한다. $\gamma$는 최적화 과정에서 실질적으로 중요합니다 (자세한 내용은 보충 자료 참조). 각각의 biLM 레이어의 활성화 함수가 서로 다른 분포를 가지고 있음을 고려할 때, 일부 경우에서 이는 가중치를 부여하기 전에 각 biLM 레이어에 layer normalization을 추가하는 데에 도움을 주었습니다 (Ba et al., 2016).

## 3.3 Using biLMs for supervised NLP tasks
&emsp;사전 학습된 biLM과 target NLP task를 위한 지도 학습 아키텍처가 주어졌을 때, task model을 개선하기 위해 biLM을 사용하는 것은 간단합니다. 우리는 간단하게 biLM을 작동시키고, 각 단어에 대한 모든 레이어의 representation을 기록하였습니다. 그리고 나서, 우리는 후술 되어 있듯이 마지막 task model이 이러한 representation의 선형 조합을 학습하도록 하였습니다.

&emsp;우선 biLM 없이 지도 학습된 모델의 최하위 레이어를 고려합니다. 대부분의 지도 학습된 NLP 모델들은 최하위 레이어에서 공통된 아키텍처를 공유하기 때문에 일관적이고 통일된 방식으로 ELMo를 추가할 수 있습니다. token으로 이루어진 서열 $(t_{1}, \dots, t_{N})$이 주어졌을 때, 각 token의 위치에 대해서 사전 학습된 단어 임베딩과 선택적으로 문자 기반의 representation을 사용하여 문맥 의존적인 token representation $\mathrm{x}_{k}$를 형성하는 것이 일반적입니다. 그리고 나서 model이 문맥에 민감한 represeentation $h_{k}$를 형성하며, 전형적으로 bidirectional RNN, CNN 또는 feed forward network가 사용됩니다.
&emsp;ELMo를 지도 학습된 모델에 추가하기 위해서, 우리는 우선 biLM의 가중치를 고정하고 ELMo vector $\mathrm{ELMo}_{k}^{task}$와 $\mathrm{x}_{k}$를 concatenate하여 ELMo enhanced representation $\left[\mathrm{x}_{k};\mathrm{ELMo}_{k}^{task}\right]$를 task RNN으로 전달합니다. 일부 task(e.g., SNLI, SQuAD)의 경우 새로운 출력 가중치를 도입하고 $h_{k}$를 $\left[h_{k};\mathrm{ELMo}_{k}^{task}\right]$로 대체하여 task RNN의 출력에 ELMo를 도입함으로써 추가적인 개선을 관측하였습니다. 지도 학습된 모델의 나머지 부분은 변경되지 않으므로 이러한 도입은 복잡한 신경망 모델의 맥락에서 일어날 수 있습니다. <font color="#ff0000">예를 들어, bi-attention layer가 biLSTM을 뒤따를 때인 Sec. 4의 SNLI 실험 또는 biLSTM의 최상위 레이어에 clustering model이 추가 되었을 때 상호 참조 해결 실험들을 보자.</font>

&emsp;최종적으로, 우리는 ELMo에 적당한 양의 드롭아웃을 추가하는 것과 (Srivastava et al., 2014) 몇몇 상황에서 ELMo의 가중치에 $\lambda ||\mathrm{w}||_{2}^2$ 을 더하여 규제하는 것이 유익하다는 것을 알아내었습니다. 이는 모든 biLM 레이어의 평균에 가깝게 위치하도록 하기 위해 ELMo의 가중치에  bias의 유도를 강제합니다.
## 3.4 Pre-trained bidirectional language model architecture
&emsp;이 논문에서 사용된 사전 학습된 biLM은 Jozefowicz et al. (2016) 및 Kim et al. (2015)과 유사하지만, 양방향의 합동 훈련을 위해 수정되었고 LSTM 레이어 사이에 residual connection을 추가하였습니다. 우리는 이 연구에서 Peters et al. (2017)이 정방향 LM과 대규모 학습에서 biLM을 사용하는 것의 중요성을 강조하였듯이, 대규모 biLM에 집중하였습니다.

&emsp;문자 기반 입력 representation을 유지할 때, 전체적인 모델의 복잡성과 모델의 크기, 다운스트림 task를 위해 요구되는 계산의 균형을 맞추기 위해 우리는 Jozefowicz et al. (2016)의 CNN-BIG-LSTM에서 모든 임베딩 및  히든 레이어의 차원을 절반으로 줄였습니다. 최종 모델은 4096의 unit과 512개의 차원 투영 그리고 첫 번째와 두 번째 레이어 사이에 residual connection을 가진 $L=2$ biLSTM입니다. c