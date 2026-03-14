# Magisterska

## Pretočni vhod in kodiranje glasbe

Zaradi pomembnosti hitrosti pri tej nalogi, poskušamo na vsakem koraku optimizirati latenco. Glede na preučene vire in sorodne modele, sem se odločil da preučim kodeke, ki spadajo v skupino RVQ - Rezidualna vektorska kavantizacija.  
To je družina nevronskih kodekov, ki zvočni signal pretvorijo v diskretne žetone poljubne kvalitete. Zaradi te lastnosti lahko izbiramo med kvaliteto in hitrostjo procesiranja

Diskretno koderianje je primerno tudi za arhitekturo modela, saj so vhodi že v obliki žetonov. Takšen vhod lahko direktno uporabimo pri modelih z transformersko arhitektruo.


### Primerjava kodekov

Primerjali smo 3 različne RVQ kodeke.

- Mimi
- DAC
- 