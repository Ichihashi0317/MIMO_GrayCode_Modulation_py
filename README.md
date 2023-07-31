# MIMO_gray_modulation_py
Python用の変復調プログラム
- BPSK, QPSK, QAM対応（すべてグレイ符号）
- 複素モデル，実数等価モデルどちらでも出力可能
### MIMO_modulatorとMassiveMIMO_modulatorの違い
- MIMO_modulatorの方が高速だが，アンテナ本数が増えるとメモリ使用量が急増するため，小規模MIMO向け
- MassiveMIMO_modulatorは，インスタンスのメモリ使用量アンテナ本数に依存せず，大規模MIMOに対応可能
- MIMO_modulatorで使用できるアンテナ本数の目安は以下の通り（PCに依存）<br>
  これを超える場合はMassiveMIMO_modulatorを使用すること
  - BPSK : 23本以下
  - QPSK : 12本以下
  - 16QAM : 6本以下
  - 64QAM : 4本以下
