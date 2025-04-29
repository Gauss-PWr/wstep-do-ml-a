#import "@preview/typslides:1.2.5": *
#import "@preview/lovelace:0.3.0": *

#show: typslides.with(
  ratio: "16-9",
  theme: "purply"
)

#front-slide(
  title: "Uczenie sieci neuronowych",
  subtitle: [Lato 2025],
  authors: "Kacper Borys"
)

#table-of-contents(title: "Spis treści")


#title-slide[
  Definicje
]

#slide(title: "Perceptron")[
 #framed(title: "Definicja - perceptron")[
    Niech $arrow(x) in bb(R)^n$. Perceptronem nazywamy funkcję $P(arrow(x)) = F(arrow(x) dot arrow(w) + b)$, gdzie
    F - funkcja aktywacji (np. sigmoidalna, ReLU, tanh), $arrow(w) in bb(R)^n, b in bb(R)$.
    ]
    Przykładowo - mamy wektor $arrow(x) = (1, 7, 3)$, wagi $arrow(w) = (0.1, 0.5, -2)$ oraz b = 1. Dla tych wartości $P (arrow(x)) = max(0, -1.4) = 0$. \ Oznacza to, że dla takich danych perceptron nie aktywuje się. 
]

#slide(title: "Warstwa")[
  #framed(title: "Definicja - warstwa")[
    Warstwą nazywamy funkcję $W(arrow(x)) = (P_1(arrow(x)), P_2(arrow(x)), dots, P_k (arrow(x)))$, gdzie $P_i$ to $i$-ty perceptron i $k in bb(N^+)$ to liczba perceptronów w warstwie.
  ]
  Kontynuując przykład. Mamy $arrow(x) = (1, 7, 3)$, wtedy dla konkretnej warstwy $W(arrow(x)) = (
    0, 0.01, 65.4, 3.1, dots.h.c, 0.422)$. Jest to nasz nowy wektor danych który możemy wykorzystać jako $arrow(x)'$ w następnej warstwie.

  #pagebreak()

  Równoważnie możemy przedstawić naszą warstwę $W(arrow(x))$ jako $F(cal(W) arrow(x) + arrow(b))$, \ #pad(top: 16pt)[gdzie $cal(W) = display(mat(
    w_(1,1), w_(1,2), dots.h.c, w_(1,n);
    w_(2,1), w_(2,2), dots.h.c, w_(2,n);
    dots.v, dots.v, dots.down, dots.v;
    w_(k,1), w_(k,2), dots.h.c, w_(k,n)
  ))$ to macierz wag, a $arrow(b) = (b_1, b_2, dots.h.c, b_k)$ to wektor przesunięcia. ]
]

#slide(title: "Sieć neuronowa")[
  #framed(title: "Definicja - sieć neuronowa")[
    Siecią neuronową nazywamy funkcję $N(arrow(x)) = arrow(y)$, gdzie $arrow(x) in bb(R)^n$ oraz $arrow(y) in bb(R)^m, m,n in bb(N)^+$. Jest ona złożeniem funkcji $W^"in" (arrow(x))$, $W^"h"_(n) (arrow(h)_(n-1))$ oraz $W^"out" (arrow(h)_n)$, gdzie $arrow(h)_0 =  W^"in" (arrow(x))$.
    Co za tym idzie $N(arrow(x)) = W^"out" (W^h_k (W^h_(k-1) (dots.h.c(W^h_1(W^"in" (arrow(x)))))))$
  ]
]

#slide(title: "Gradient")[
  #framed(title: "Definicja - gradient")[
    Gradientem funkcji $f$ w punkcie $x$ nazywamy wektor $nabla f(x) = ( (partial f)/ (partial x_1), (partial f)/(partial x_2), dots.h.c, (partial f)/(partial x_n))$.
  ]
  Mająć jakąś funkcję która przyjmuje wektor, możemy zastosować na niej gradient - zróżniczkować ją względem każdego z parametrów. W ten sposób uzyskujemy wektor, który pokazuje w którą stronę powinniśmy iść aby zwiększyć wartość funkcji. \ #pad(top: 16pt)[Przykładowo - mamy funkcję $f(x) = x^2 + 2x + 1$, wtedy $nabla f(x) = (2x + 2)$, a dla $x = 1$ mamy $nabla f(1) = 4$. Czyli jeśli jesteśmy w $x = 1$ to powinniśmy iść w prawo na osi liczbowej.]


  #framed(title: "Uwaga")[
    Gradient wskazuje *kierunek* w którym powinniśmy się poruszać. Długość wektora obrazuje *szybkość* zmiany funkcji.
  ]

  #pagebreak()

  #set align(center)
  #image("gradient_ascent.gif")
]


#title-slide[
  Uczenie sieci
]

#slide(title: "Przykład - wstęp")[
  Niech $p(x) = a x^2 + b x + c$. Aby uzyskać wartość $p(1) = a+b+c = 2$ musimy dobrać odpowiednie parametry. Możemy wyliczyć je analitycznie, ale możemy również wykorzystać gradient.

  #pagebreak()

  Skoro gradient wskazuje kierunek największego wzrostu funkcji, to możemy iść w przeciwną stronę aby zmniejszyć wartość funkcji, innymi słowy - minimalizować ją. 

  To co chcemy minimalizować to $L(a,b,c) = (2 - p(a, b, c))^2$ - nazywamy to funkcją kosztu. Pokazuje nam ona jak bardzo funkcja oddalona jest od zadanej wartości.

  Przykładowo - dobierzmy parametry $a = 1, b = 1, c = 0$, wtedy $p(1) = 2$, a funkcja kosztu wynosi $0$. Jeśli dobierzemy $a = 1, b = 1, c = 1$, wtedy $p(1) = 3$, a funkcja kosztu wynosi $1$. 
  
  Gdy nałożymy gradient na funkcję kosztu i zmieniemy znak otrzymamy wektor który wskazuje na największy spadek funkcji.
  
]

#slide(title: "Przykład - gradient")[
  Obliczmy gradient funkcji kosztu.
  #align(center)[
  $nabla L(a,b,c) = ((partial L)/(partial a), (partial L)/(partial b), (partial L)/(partial c))$
  ]
  Korzystając z reguły łańcuchowej mamy:
  #align(center)[
  $nabla L(a,b,c) = 2(p(a,b,c) - 2)(nabla p(a,b,c))$
  ]
]

#slide(title: "Przykład - gradient decent")[
  Skoro mamy gradient, to możemy go wykorzystać do zmiany parametrów.
  Zbudujmy algorytm który będzie aktualizował nasze parametry:
  #pseudocode-list[
    + zainicuj parametry $a,b,c$
    + zainicjuj krok $alpha$
    + zainicjuj $epsilon$
    + *dopóki* $L(a,b,c) > epsilon$
      + oblicz gradient $nabla L(a,b,c)$
      + (a, b, c) = (a, b, c) - $alpha * nabla L(a,b,c)$
    + *koniec*
  ]

  #set align(center)
  #image("Gradient_descent.gif", format: "gif")
]

#slide(title: "Propagacja wsteczna")[
  #framed(title: "Definicja - propagacja wsteczna")[
    Propagacją wsteczną nazywamy proces aktualizowania parametrów sieci neuronowej przy pomocy $nabla L(arrow(y) - N(arrow(p)))$  gdzie $arrow(p)$ jest wektorem złożonym z wag oraz biasów.
  ]
  Znając gradient $W^"out"$, dzięki regule łańcuchowej, możemy obliczyć gradient $W^"h"_(n)$, a znając gradient $W^"h"_(n)$ możemy obliczyć gradient $W^"h"_(n-1)$ i tak dalej aż do $W^"in"$. Tym sposobem możemy uczyć naszą sieć neuronową.

  #pagebreak()

  Te równania dokładnie opisują algorytm propagacji wstecznej:\
  #align(center)[
    $
      delta^"out" &= nabla_sigma(arrow(h))L dot.circle sigma'(arrow(h)) \
      delta^"h"_(n-1) &= (W_n)^T delta_n dot.circle sigma'(arrow(h)_(n)) \
      delta_(n) &= (partial L)/(partial arrow(b)_n) \
      sigma(arrow(h)_(n-1))delta_n &= (partial L)/(partial arrow(w)_n) 
    $
  ]
  Gdzie $dot.circle$ jest mnożeniem Hadamarda (_element-wise_), $sigma$ jest funkcją aktywacji, a $L$ jest funkcją kosztu.\
]