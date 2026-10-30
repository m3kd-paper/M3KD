-----------------500step----------------------------

when llama is judge
Position-invariant DistiLLM preference: 122/133 (91.73%)
Position-invariant SFT preference: 11/133 (8.27%)

Position-sensitive cases: 64/200 (32.00%)
  DistiLLM only when B: 42
  DistiLLM only when A: 22
  B-vs-A asymmetry: +20 cases




when qwen is judge
Position-invariant DistiLLM preference: 49/57 (85.96%)
Position-invariant SFT preference: 8/57 (14.04%)

Position-sensitive cases: 50/200 (25.00%)
  DistiLLM only when B: 17
  DistiLLM only when A: 33
  B-vs-A asymmetry: -16 cases


-----------------1000step---temp0.8-------------------------


--------------------------------------------------------------------------------
Position-invariant DistiLLM preference: 125/132 (94.70%)
Position-invariant SFT preference: 7/132 (5.30%)

Position-sensitive cases: 59/200 (29.50%)
  DistiLLM only when B: 33
  DistiLLM only when A: 26
  B-vs-A asymmetry: +7 cases

================================================================================
RAW VERDICT PAIRS
================================================================================
?->A                                    :   1/200 (  0.50%)
A->A                                    :  26/200 ( 13.00%)
A->B                                    :   7/200 (  3.50%)
B->A                                    : 125/200 ( 62.50%)
B->B                                    :  33/200 ( 16.50%)
B->C                                    :   4/200 (  2.00%)
C->A                                    :   3/200 (  1.50%)
C->B                                    :   1/200 (  0.50%)






-----------------------------1000step---------temp0------------------------------------------
Position-invariant DistiLLM preference: 143/149 (95.97%)
Position-invariant SFT preference: 6/149 (4.03%)

Position-sensitive cases: 43/200 (21.50%)
  DistiLLM only when B: 28
  DistiLLM only when A: 15
  B-vs-A asymmetry: +13 cases

================================================================================
RAW VERDICT PAIRS
================================================================================
A->A                                    :  15/200 (  7.50%)
A->B                                    :   6/200 (  3.00%)
A->C                                    :   1/200 (  0.50%)
B->A                                    : 143/200 ( 71.50%)
B->B                                    :  28/200 ( 14.00%)
B->C                                    :   2/200 (  1.00%)
C->A                                    :   3/200 (  1.50%)
C->B                                    :   1/200 (  0.50%)
C->C                                    :   1/200 (  0.50%)




----------------2000step---------------------------
--------------------------------------------------------------------------------
Position-invariant DistiLLM preference: 127/136 (93.38%)
Position-invariant SFT preference: 9/136 (6.62%)

Position-sensitive cases: 59/200 (29.50%)
  DistiLLM only when B: 34
  DistiLLM only when A: 25
  B-vs-A asymmetry: +9 cases

================================================================================
RAW VERDICT PAIRS
================================================================================
?->B                                    :   1/200 (  0.50%)
A->A                                    :  25/200 ( 12.50%)
A->B                                    :   9/200 (  4.50%)
B->A                                    : 127/200 ( 63.50%)
B->B                                    :  34/200 ( 17.00%)
B->C                                    :   2/200 (  1.00%)
C->A                                    :   1/200 (  0.50%)
C->B                                    :   1/200 (  0.50%)