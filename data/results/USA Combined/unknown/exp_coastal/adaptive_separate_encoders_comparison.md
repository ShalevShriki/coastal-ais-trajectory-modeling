# Separate-encoder adaptive gate comparison

| Model | Gate | Median FDE | Mean FDE | Median ADE | Mean ADE | Most Selected | Selection % | Gate H |
|-------|------|------------|----------|------------|----------|---------------|-------------|--------|
| Separate Encoders | Softmax | 21.00 | 37.70 | 12.38 | 19.22 | 9h | 99.1% | 1.179 |
| Separate Encoders | Hard | 19.08 | 36.90 | 11.38 | 19.07 | 12h | 36.8% | -0.000 |

## Representation cosine (mean hidden states)

- **Softmax**: cos(9,12)=0.271, cos(12,18)=0.108, cos(18,24)=0.046
- **Hard**: cos(9,12)=0.293, cos(12,18)=0.140, cos(18,24)=0.104

## Alpha / selection breakdown

### Softmax
- mean α: {'9h': 0.49274665117263794, '12h': 0.0948396623134613, '18h': 0.2597334086894989, '24h': 0.15268027782440186}
- argmax %: {'9h': 99.05225549555446, '12h': 0.03000054546446299, '18h': 0.9163802978235969, '24h': 0.0013636611574755905}
- softmax entropy (pre-hard): 1.179

### Hard
- mean α: {'9h': 0.291496217250824, '12h': 0.3683794140815735, '18h': 0.0, '24h': 0.34012436866760254}
- argmax %: {'9h': 29.14962090219822, '12h': 36.8379425080456, '18h': 0.0, '24h': 34.01243658975618}
- softmax entropy (pre-hard): 0.039

