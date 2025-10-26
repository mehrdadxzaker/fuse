# Symbolic Gradient Programs

## MLP Dot-Product Loss

```fuse
Input[b,d]      = const([[0.8,0.2,0.5], [0.1,0.6,0.4]])
W1[h,d]         = const([[0.3,-0.2,0.5], [0.7,0.1,-0.4]])
B1[h]           = const([0.1,-0.2])
W2[o,h]         = const([[0.2,0.6], [-0.5,0.3]])
B2[o]           = const([0.0,0.1])
Target[b,o]     = const([[0.4,0.6], [0.3,0.7]])
BatchOnes[b]    = const([1.0, 1.0])

HiddenLinear[b,h] = W1[h,d] Input[b,d]
HiddenBias[b,h]   = BatchOnes[b] B1[h]
Hidden[b,h]       = HiddenLinear[b,h]
Hidden[b,h]       = HiddenBias[b,h]
Activation[b,h]   = gelu(Hidden[b,h])
Logits[b,o]       = W2[o,h] Activation[b,h]
OutputBias[b,o]   = BatchOnes[b] B2[o]
Scores[b,o]       = Logits[b,o]
Scores[b,o]       = OutputBias[b,o]
Probs[b,o]        = softmax(Scores[b,o], axis="o")
Loss              = Probs[b,o] Target[b,o]

Grad_Loss = const(1.0)
Grad_Probs[b, o] = Grad_Loss Target[b, o]
Grad_Target[b, o] = Grad_Loss Probs[b, o]
Grad_Scores[b, o] = softmax_grad(Probs[b, o], Grad_Probs[b, o], axis="o")
Grad_OutputBias[b, o] = Grad_Scores[b, o]
Grad_Logits[b, o] = Grad_Scores[b, o]
Grad_BatchOnes[b] = Grad_OutputBias[b, o] B2[o]
Grad_B2[o] = Grad_OutputBias[b, o] BatchOnes[b]
Grad_W2[o, h] = Grad_Logits[b, o] Activation[b, h]
Grad_Activation[b, h] = Grad_Logits[b, o] W2[o, h]
Grad_Hidden[b, h] = Grad_Activation[b, h] gelu_grad(Hidden[b, h])
Grad_HiddenBias[b, h] = Grad_Hidden[b, h]
Grad_HiddenLinear[b, h] = Grad_Hidden[b, h]
Grad_BatchOnes[b] = Grad_HiddenBias[b, h] B1[h]
Grad_B1[h] = Grad_HiddenBias[b, h] BatchOnes[b]
Grad_W1[h, d] = Grad_HiddenLinear[b, h] Input[b, d]
Grad_Input[b, d] = Grad_HiddenLinear[b, h] W1[h, d]
```

## Attention Dot-Product Loss

```fuse
X[p,d]       = const([[0.1,0.2,0.3],[0.4,0.5,0.6]])
WQ[dk,d]     = const([[0.2,0.1,0.0],[0.0,0.3,0.1],[0.1,-0.2,0.2]])
WK[dk,d]     = const([[0.1,0.4,0.2],[0.3,0.2,0.1],[0.0,0.1,0.3]])
WV[dv,d]     = const([[0.5,0.4,0.3],[0.2,0.1,0.0],[0.3,0.2,0.1]])
Mask[p,p']   = const([[0.0,-1.0],[-1.0,0.0]])
InvSqrtDk    = const(0.35355339)
Target[p,dv] = const([[0.1,0.2,0.3],[0.3,0.2,0.1]])

Q[p,dk]      = WQ[dk,d] X[p,d]
K[p,dk]      = WK[dk,d] X[p,d]
V[p,dv]      = WV[dv,d] X[p,d]
Score[p,p']  = Q[p,dk] K[p',dk] InvSqrtDk
Score[p,p']  = Mask[p,p']
Comp[p,p'.]  = softmax(Score[p,p'], axis="p'")
Attn[p,dv]   = Comp[p,p'] V[p',dv]
Loss         = Attn[p,dv] Target[p,dv]

Grad_Loss = const(1.0)
Grad_Attn[p, dv] = Grad_Loss Target[p, dv]
Grad_Target[p, dv] = Grad_Loss Attn[p, dv]
Grad_Comp[p, p'] = Grad_Attn[p, dv] V[p', dv]
Grad_V[p', dv] = Grad_Attn[p, dv] Comp[p, p']
Grad_Score[p, p'] = softmax_grad(Comp[p, p'], Grad_Comp[p, p'], axis="p'")
Grad_Mask[p, p'] = Grad_Score[p, p']
Grad_Q[p, dk] = Grad_Score[p, p'] K[p', dk] InvSqrtDk
Grad_K[p', dk] = Grad_Score[p, p'] Q[p, dk] InvSqrtDk
Grad_InvSqrtDk = Grad_Score[p, p'] Q[p, dk] K[p', dk]
Grad_WV[dv, d] = Grad_V[p, dv] X[p, d]
Grad_X[p, d] = Grad_V[p, dv] WV[dv, d]
Grad_WK[dk, d] = Grad_K[p, dk] X[p, d]
Grad_X[p, d] = Grad_K[p, dk] WK[dk, d]
Grad_WQ[dk, d] = Grad_Q[p, dk] X[p, d]
Grad_X[p, d] = Grad_Q[p, dk] WQ[dk, d]
```
