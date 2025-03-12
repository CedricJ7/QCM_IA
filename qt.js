// questions-theoriques.js
// 30 Questions théoriques sur l'IA

const questionsTheoriques = [
    // Questions originales
    {
        question: "Dans quelle situation l'utilisation d'un réseau de neurones récurrent (RNN) est-elle plus appropriée qu'un CNN ?",
        options: [
            "Pour l'analyse d'images satellites",
            "Pour le traitement de séquences temporelles comme les séries chronologiques ou le texte",
            "Pour la classification d'images médicales",
            "Pour la segmentation sémantique d'images"
        ],
        correctIndex: 1,
        explanation: "Les RNN sont spécialement conçus pour traiter des données séquentielles où l'ordre temporel est important, comme le texte, les séries temporelles ou l'audio. Ils possèdent des connexions récurrentes qui permettent de maintenir un état interne (mémoire) capturant les dépendances temporelles. Les CNN, en revanche, excellent dans l'extraction de caractéristiques spatiales locales et hiérarchiques dans des données avec une structure en grille, comme les images.",
        difficulty: "medium"
    },
    {
        question: "Quelle est la différence fondamentale entre les opérations de pooling et de convolution dans un CNN ?",
        options: [
            "Le pooling extrait des caractéristiques, tandis que la convolution réduit la dimensionnalité",
            "Le pooling applique une fonction non-linéaire, tandis que la convolution est toujours linéaire",
            "La convolution extrait des caractéristiques via des filtres apprenables, tandis que le pooling réduit la dimensionnalité sans paramètres apprenables",
            "La convolution fonctionne uniquement sur les images en niveaux de gris, tandis que le pooling fonctionne sur les images couleur"
        ],
        correctIndex: 2,
        explanation: "La convolution utilise des filtres avec des poids apprenables pour extraire des caractéristiques spécifiques (comme des contours, textures, etc.) en effectuant une opération de produit scalaire entre le filtre et les régions de l'entrée. Le pooling, en revanche, est une opération de sous-échantillonnage sans paramètres apprenables qui réduit la dimensionnalité spatiale en agrégeant l'information localement (par exemple, en prenant le maximum ou la moyenne d'une région).",
        difficulty: "easy"
    },
    {
        question: "Dans l'algorithme AdaBoost, comment sont pondérés les échantillons après chaque itération ?",
        options: [
            "Les poids de tous les échantillons sont uniformément diminués",
            "Les poids des échantillons correctement classifiés sont augmentés, ceux mal classifiés sont diminués",
            "Les poids des échantillons correctement classifiés sont diminués, ceux mal classifiés sont augmentés",
            "Les poids restent constants, seuls les poids des classificateurs faibles changent"
        ],
        correctIndex: 2,
        explanation: "Dans AdaBoost, après chaque itération, les poids des échantillons correctement classifiés sont diminués tandis que ceux des échantillons mal classifiés sont augmentés. Cela force les classificateurs faibles suivants à se concentrer davantage sur les exemples difficiles (mal classifiés), améliorant ainsi progressivement la performance de l'ensemble. La formule mathématique utilisée est Dt+1(i) = Dt(i) × exp(-αt × yi × ht(xi)) où αt est le poids du classificateur, yi est l'étiquette réelle et ht(xi) est la prédiction.",
        difficulty: "medium"
    },
    {
        question: "Quelles techniques permettent de résoudre efficacement le problème d'explosion du gradient dans les RNN ?",
        options: [
            "L'augmentation du taux d'apprentissage et l'utilisation d'un plus grand nombre de couches",
            "L'utilisation exclusive de fonctions d'activation linéaires",
            "L'écrêtage du gradient (gradient clipping) et l'utilisation d'architectures comme LSTM ou GRU",
            "Le suréchantillonnage des données d'entraînement et la normalisation par lots"
        ],
        correctIndex: 2,
        explanation: "L'explosion du gradient est particulièrement problématique dans les RNN profonds. Deux principales solutions sont : (1) l'écrêtage du gradient (gradient clipping) qui limite la norme du gradient à un seuil maximum, empêchant ainsi des mises à jour trop importantes, et (2) l'utilisation d'architectures spéciales comme LSTM (Long Short-Term Memory) ou GRU (Gated Recurrent Unit) qui sont conçues avec des mécanismes de portes pour mieux contrôler le flux du gradient à travers le temps.",
        difficulty: "medium"
    },
    {
        question: "Pourquoi RMSE (Root Mean Squared Error) est-elle souvent préférée à MSE comme métrique d'évaluation ?",
        options: [
            "RMSE est toujours plus petite que MSE, donc plus facile à interpréter",
            "RMSE est exprimée dans la même unité que la variable cible, ce qui facilite l'interprétation",
            "RMSE est moins sensible aux valeurs aberrantes que MSE",
            "RMSE converge plus rapidement que MSE lors de l'optimisation"
        ],
        correctIndex: 1,
        explanation: "RMSE est la racine carrée de MSE (RMSE = √MSE). Son principal avantage est qu'elle est exprimée dans la même unité que la variable prédite, ce qui facilite l'interprétation des erreurs. Par exemple, si on prédit des prix en euros, MSE serait en euros², tandis que RMSE serait en euros. Mathématiquement, les deux métriques sont monotoniquement liées, donc minimiser l'une revient à minimiser l'autre.",
        difficulty: "easy"
    },
    {
        question: "Quelle fonction de coût est la plus appropriée pour un problème de classification binaire et pourquoi ?",
        options: [
            "MSE (Mean Squared Error), car elle pénalise plus fortement les erreurs importantes",
            "MAE (Mean Absolute Error), car elle est plus robuste aux valeurs aberrantes",
            "Entropie croisée binaire (Binary Cross-Entropy), car elle est spécifiquement conçue pour les probabilités et pénalise fortement les prédictions confiantes mais incorrectes",
            "L'erreur de Poisson, car elle est adaptée aux distributions binomiales"
        ],
        correctIndex: 2,
        explanation: "L'entropie croisée binaire (BCE) est la fonction de coût la plus adaptée pour la classification binaire car elle mesure la divergence entre deux distributions de probabilité : les étiquettes réelles et les prédictions. Mathématiquement, elle est définie comme BCE = -Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]. Sa principale force est qu'elle pénalise très fortement les prédictions confiantes mais incorrectes (ex. prédire 0.01 quand la vraie valeur est 1), ce qui incite le modèle à être prudent dans ses prédictions.",
        difficulty: "medium"
    },
    
    // Nouvelles questions théoriques
    {
        question: "Qu'est-ce que le vanishing gradient problem (problème de disparition du gradient) et pourquoi est-il important dans les réseaux de neurones profonds ?",
        options: [
            "C'est un problème où les gradients deviennent trop grands et font diverger l'entraînement",
            "C'est un problème où les gradients deviennent extrêmement petits dans les couches initiales, ralentissant ou bloquant l'apprentissage",
            "C'est un problème où les gradients sont calculés incorrectement par l'algorithme de backpropagation",
            "C'est un problème où la fonction de coût ne converge pas vers un minimum global"
        ],
        correctIndex: 1,
        explanation: "Le problème de disparition du gradient (vanishing gradient) se produit lorsque les gradients deviennent exponentiellement petits en se propageant vers les couches initiales d'un réseau profond. Ce phénomène est particulièrement problématique car il empêche les poids des premières couches d'être mis à jour efficacement, ce qui ralentit considérablement l'apprentissage ou peut même le bloquer complètement. Il est principalement causé par l'utilisation de certaines fonctions d'activation (comme sigmoid ou tanh) dont les dérivées sont proches de zéro pour des valeurs d'entrée éloignées de l'origine, et par la multiplication répétée de ces petites valeurs lors de la backpropagation à travers de nombreuses couches.",
        difficulty: "medium"
    },
    {
        question: "Quelle fonction d'activation aide à atténuer le problème de vanishing gradient dans les réseaux de neurones profonds ?",
        options: [
            "Sigmoid",
            "Tanh",
            "ReLU (Rectified Linear Unit)",
            "Softmax"
        ],
        correctIndex: 2,
        explanation: "La fonction d'activation ReLU (Rectified Linear Unit), définie par f(x) = max(0, x), aide à atténuer le problème de vanishing gradient car sa dérivée est soit 0 (pour x < 0) soit 1 (pour x > 0). Contrairement aux fonctions sigmoid et tanh dont les dérivées s'approchent de zéro pour des valeurs éloignées de l'origine, la dérivée de ReLU ne s'atténue pas pour les valeurs positives, ce qui permet aux gradients de se propager plus efficacement à travers les couches profondes. Cependant, ReLU peut souffrir du problème des 'neurones morts' (lorsque les neurones restent inactifs pour toutes les entrées), ce qui a conduit à des variantes comme Leaky ReLU, PReLU, et ELU.",
        difficulty: "medium"
    },
    {
        question: "Quelles sont les principales stratégies pour traiter les valeurs manquantes dans un jeu de données ?",
        options: [
            "Suppression des observations ou des variables ayant des valeurs manquantes",
            "Imputation par la moyenne, la médiane, le mode ou des méthodes plus avancées",
            "Utilisation de modèles qui peuvent gérer nativement les valeurs manquantes",
            "Toutes les réponses précédentes"
        ],
        correctIndex: 3,
        explanation: "Les stratégies pour traiter les valeurs manquantes comprennent : (1) la suppression des lignes (observations) ou colonnes (variables) contenant des valeurs manquantes, ce qui est simple mais peut entraîner une perte d'information ; (2) l'imputation, qui consiste à remplacer les valeurs manquantes par des estimations comme la moyenne, la médiane, le mode, ou via des méthodes plus sophistiquées comme l'imputation par k plus proches voisins, par régression, ou par algorithmes d'apprentissage automatique ; (3) l'utilisation de modèles qui peuvent gérer nativement les valeurs manquantes, comme les arbres de décision qui peuvent traiter les données manquantes en utilisant des règles de substitution ou en les incorporant dans leur processus de division.",
        difficulty: "easy"
    },
    {
        question: "Comment fonctionne l'algorithme des k plus proches voisins (k-NN) ?",
        options: [
            "Il divise récursivement l'espace des caractéristiques en régions et attribue une classe à chaque région",
            "Il apprend un ensemble de poids qui minimisent une fonction de coût sur les données d'entraînement",
            "Il classe un nouvel exemple en fonction des k exemples d'entraînement les plus proches dans l'espace des caractéristiques",
            "Il utilise des transformations en ondelettes pour identifier les motifs récurrents dans les données"
        ],
        correctIndex: 2,
        explanation: "L'algorithme des k plus proches voisins (k-NN) est une méthode d'apprentissage supervisé non paramétrique qui classifie un nouvel exemple en se basant sur les k exemples d'entraînement les plus proches dans l'espace des caractéristiques. Pour prédire la classe d'un nouvel exemple, l'algorithme : (1) calcule la distance (généralement euclidienne) entre ce nouvel exemple et tous les exemples d'entraînement ; (2) identifie les k plus proches voisins (ceux ayant les distances les plus faibles) ; (3) attribue la classe majoritaire parmi ces k voisins (pour la classification) ou calcule la moyenne de leurs valeurs (pour la régression). k-NN est conceptuellement simple et n'a pas de phase d'entraînement explicite, mais peut être computationnellement coûteux pour de grands ensembles de données.",
        difficulty: "easy"
    },
    {
        question: "Quels sont les principaux avantages et inconvénients de l'algorithme k-NN ?",
        options: [
            "Avantages : simplicité, pas d'hypothèse sur les données, s'adapte aux frontières de décision complexes ; Inconvénients : coût computationnel élevé, sensible aux caractéristiques non pertinentes, nécessite une normalisation des données",
            "Avantages : rapidité d'exécution, faible utilisation de la mémoire ; Inconvénients : difficulté à capturer des frontières de décision non linéaires",
            "Avantages : excellent traitement des valeurs aberrantes, performances optimales sur les données de grande dimension ; Inconvénients : nécessite beaucoup de données d'entraînement",
            "Avantages : capacité à apprendre les représentations hiérarchiques ; Inconvénients : surapprentissage fréquent, difficulté d'interprétation"
        ],
        correctIndex: 0,
        explanation: "L'algorithme k-NN présente plusieurs avantages : il est simple à comprendre et à implémenter, ne fait aucune hypothèse sur la distribution des données, et peut modéliser des frontières de décision complexes et non linéaires. Cependant, ses inconvénients sont significatifs : (1) son coût computationnel est élevé, surtout avec de grands jeux de données, car il nécessite de calculer les distances avec tous les exemples d'entraînement ; (2) il est très sensible aux caractéristiques non pertinentes ou redondantes, car elles influencent directement le calcul des distances ; (3) l'algorithme nécessite généralement une normalisation des données pour éviter que les caractéristiques avec de grandes échelles ne dominent le calcul des distances ; (4) le choix du paramètre k et de la métrique de distance est crucial et peut affecter significativement les performances.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que la régression logistique, malgré son nom ?",
        options: [
            "Un algorithme de régression pour prédire des valeurs numériques continues",
            "Un algorithme de classification qui estime la probabilité qu'une instance appartienne à une certaine classe",
            "Une méthode de réduction de dimensionnalité similaire à l'analyse en composantes principales",
            "Une technique de clustering qui maximise la séparation entre les groupes"
        ],
        correctIndex: 1,
        explanation: "Malgré son nom suggérant un algorithme de régression, la régression logistique est fondamentalement un algorithme de classification binaire (bien qu'elle puisse être étendue à la classification multi-classes). Elle estime la probabilité qu'une instance appartienne à une certaine classe en appliquant une fonction logistique (sigmoïde) à une combinaison linéaire des caractéristiques. La fonction sigmoïde transforme la sortie pour qu'elle soit comprise entre 0 et 1, ce qui peut être interprété comme une probabilité. Si cette probabilité dépasse un certain seuil (généralement 0,5), l'instance est classée dans la classe positive, sinon dans la classe négative. La régression logistique est largement utilisée en raison de sa simplicité, de son interprétabilité et de son efficacité computationnelle.",
        difficulty: "easy"
    },
    {
        question: "Comment la régression logistique transforme-t-elle son résultat en probabilité ?",
        options: [
            "En utilisant une fonction exponentielle qui amplifie les valeurs positives",
            "En normalisant les résultats par la somme de toutes les prédictions possibles",
            "En appliquant une fonction sigmoïde (logistique) à la combinaison linéaire des caractéristiques",
            "En transformant les résultats à l'aide d'une tangente hyperbolique"
        ],
        correctIndex: 2,
        explanation: "La régression logistique transforme une combinaison linéaire des caractéristiques (z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ) en une probabilité en appliquant la fonction sigmoïde (aussi appelée fonction logistique), définie par σ(z) = 1/(1+e^(-z)). Cette fonction mappe n'importe quelle valeur réelle à un nombre entre 0 et 1, ce qui peut être interprété comme une probabilité. La forme en S de la sigmoïde est particulièrement adaptée car elle capture bien la transition entre les classes : elle tend vers 0 pour les valeurs très négatives, vers 1 pour les valeurs très positives, et varie plus rapidement autour de 0. L'entraînement du modèle consiste à trouver les poids w qui maximisent la vraisemblance des données observées, généralement en minimisant la fonction de perte d'entropie croisée binaire.",
        difficulty: "medium"
    },
    {
        question: "Quelle est la différence entre une validation croisée k-fold et une validation avec ensemble de test fixe ?",
        options: [
            "La validation croisée k-fold est plus rapide mais moins précise que la validation avec ensemble de test fixe",
            "La validation croisée k-fold utilise toutes les données pour l'entraînement et le test, ce qui fournit une estimation plus robuste de la performance du modèle",
            "La validation avec ensemble de test fixe permet d'éviter le sur-apprentissage, contrairement à la validation croisée k-fold",
            "La validation croisée k-fold est uniquement adaptée aux problèmes de classification, tandis que la validation avec ensemble de test fixe convient à tout type de problème"
        ],
        correctIndex: 1,
        explanation: "La principale différence réside dans l'utilisation des données. Dans la validation avec ensemble de test fixe, on divise une seule fois les données en ensembles d'entraînement et de test, puis on évalue le modèle sur cet ensemble de test unique. Cette approche est simple mais sensible à la répartition spécifique choisie. La validation croisée k-fold, en revanche, divise les données en k sous-ensembles (folds) et effectue k itérations, chaque sous-ensemble servant une fois comme ensemble de test tandis que les autres servent à l'entraînement. Les performances sont moyennées sur les k itérations, ce qui fournit une estimation plus robuste car chaque exemple est utilisé à la fois pour l'entraînement et le test. Cette méthode utilise donc plus efficacement les données disponibles, surtout lorsqu'elles sont limitées, mais elle est computationnellement plus coûteuse puisqu'elle nécessite d'entraîner k modèles distincts.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que la technique de dropout dans les réseaux de neurones ?",
        options: [
            "Une méthode pour retirer progressivement les couches d'un réseau pendant l'entraînement",
            "Une technique d'initialisation des poids qui élimine les connexions inutiles",
            "Une technique de régularisation qui désactive aléatoirement certains neurones pendant l'entraînement",
            "Une stratégie d'arrêt précoce qui interrompt l'entraînement quand la performance commence à se dégrader"
        ],
        correctIndex: 2,
        explanation: "Le dropout est une technique de régularisation puissante pour les réseaux de neurones, introduite pour combattre le surapprentissage. Pendant l'entraînement, à chaque itération, certains neurones (et leurs connexions) sont temporairement désactivés (\"dropped out\") avec une probabilité p (typiquement 0.5 pour les couches cachées). Cela force le réseau à apprendre des représentations plus robustes car il ne peut pas trop s'appuyer sur des neurones spécifiques. En effet, chaque mise à jour des poids se fait dans un sous-réseau aléatoirement \"élagué\". Lors de l'inférence, tous les neurones sont activés, mais leurs sorties sont multipliées par (1-p) pour compenser le fait qu'il y a plus de neurones actifs qu'en moyenne pendant l'entraînement. Cette technique s'apparente à faire une moyenne d'ensemble implicite de nombreux sous-réseaux, ce qui améliore la généralisation.",
        difficulty: "medium"
    },
    {
        question: "Quel est le rôle de la fonction d'activation dans un réseau de neurones ?",
        options: [
            "Elle normalise les entrées du réseau pour faciliter l'apprentissage",
            "Elle introduit de la non-linéarité, permettant au réseau de modéliser des relations complexes",
            "Elle régularise le réseau pour éviter le surapprentissage",
            "Elle accélère la convergence de l'algorithme d'optimisation"
        ],
        correctIndex: 1,
        explanation: "La fonction d'activation joue un rôle crucial en introduisant de la non-linéarité dans un réseau de neurones. Sans fonctions d'activation non linéaires, un réseau de neurones, quelle que soit sa profondeur, serait équivalent à une simple régression linéaire, car la composition de plusieurs transformations linéaires reste une transformation linéaire. Cette non-linéarité permet au réseau d'apprendre des relations complexes et de modéliser pratiquement n'importe quelle fonction, conformément au théorème d'approximation universelle. Les fonctions d'activation courantes incluent ReLU (max(0,x)), qui est simple et efficace pour les réseaux profonds, sigmoid (1/(1+e^(-x))) qui mappe les sorties entre 0 et 1, utile pour les probabilités, et tanh, similaire à sigmoid mais centrée autour de 0, souvent utilisée dans les RNN.",
        difficulty: "easy"
    },
    {
        question: "Comment fonctionne l'optimiseur Adam par rapport à la descente de gradient stochastique (SGD) classique ?",
        options: [
            "Adam utilise un taux d'apprentissage fixe, tandis que SGD adapte le taux d'apprentissage",
            "Adam calcule des gradients de second ordre pour trouver des minima plus rapidement",
            "Adam maintient des taux d'apprentissage individuels pour chaque paramètre et combine les avantages de RMSprop et Momentum",
            "Adam est spécifiquement conçu pour les réseaux de neurones convolutifs, contrairement à SGD qui est générique"
        ],
        correctIndex: 2,
        explanation: "Adam (Adaptive Moment Estimation) est un algorithme d'optimisation qui combine les avantages de deux autres extensions de la descente de gradient stochastique : RMSprop et Momentum. Contrairement au SGD classique qui utilise un taux d'apprentissage unique pour tous les paramètres, Adam maintient un taux d'apprentissage individuel pour chaque paramètre du réseau. Il calcule des estimations adaptatives des moments de premier ordre (la moyenne, similaire au momentum) et de second ordre (la variance non centrée, similaire à RMSprop) des gradients. Cela permet à Adam d'adapter dynamiquement le taux d'apprentissage pour chaque paramètre en fonction de son historique récent de gradients, ce qui accélère généralement la convergence et peut aider à sortir des plateaux et des minima locaux. Adam est souvent le choix par défaut dans de nombreuses applications de deep learning en raison de sa robustesse et de son efficacité.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que la validation croisée stratifiée et quand est-elle particulièrement utile ?",
        options: [
            "Une méthode pour équilibrer le nombre d'échantillons dans chaque classe, utile pour les données très déséquilibrées",
            "Une technique qui préserve la proportion des classes dans chaque fold, utile pour les données déséquilibrées",
            "Une approche qui stratifie les hyperparamètres pour optimiser les performances, utile pour les modèles complexes",
            "Une méthode de validation qui sépare les données par caractéristiques, utile pour les données de grande dimension"
        ],
        correctIndex: 1,
        explanation: "La validation croisée stratifiée est une variante de la validation croisée k-fold qui s'assure que chaque fold contient approximativement la même proportion de classes que l'ensemble de données original. Cette méthode est particulièrement utile pour les jeux de données déséquilibrés, où certaines classes sont beaucoup moins représentées que d'autres. Sans stratification, on risque d'avoir des folds où certaines classes sont surreprésentées ou sous-représentées, voire absentes, ce qui peut biaiser l'estimation de performance du modèle. Par exemple, si une classe minoritaire ne représente que 5% des données, elle pourrait être complètement absente de certains folds dans une validation croisée standard. La stratification garantit que chaque classe est représentée de manière proportionnelle dans chaque fold, ce qui donne une évaluation plus fiable et moins variable du modèle.",
        difficulty: "medium"
    },
    {
        question: "Pourquoi normalise-t-on généralement les données d'entrée en apprentissage automatique ?",
        options: [
            "Pour s'assurer que toutes les caractéristiques contribuent également à la fonction de coût",
            "Pour accélérer la convergence des algorithmes d'optimisation basés sur le gradient",
            "Pour améliorer la stabilité numérique et éviter les problèmes liés aux différences d'échelle",
            "Toutes les réponses précédentes"
        ],
        correctIndex: 3,
        explanation: "La normalisation des données d'entrée est cruciale pour de nombreuses raisons : (1) Elle garantit que toutes les caractéristiques contribuent équitablement à la fonction de coût, évitant que les caractéristiques avec de grandes valeurs ne dominent celles avec de petites valeurs. (2) Elle accélère significativement la convergence des algorithmes d'optimisation basés sur le gradient. Sans normalisation, le gradient peut osciller fortement dans les directions correspondant aux caractéristiques de grande amplitude, ce qui nécessite un taux d'apprentissage plus faible et ralentit l'apprentissage. (3) Elle améliore la stabilité numérique et évite les problèmes de précision, particulièrement importants dans les architectures profondes. De plus, certains algorithmes, comme les SVM avec noyaux RBF ou k-NN, sont intrinsèquement sensibles aux échelles et nécessitent une normalisation pour fonctionner correctement. Les méthodes courantes incluent la normalisation min-max (mise à l'échelle entre 0 et 1) et la standardisation (centrage-réduction, pour obtenir une moyenne de 0 et un écart-type de 1).",
        difficulty: "easy"
    },
    {
        question: "Comment fonctionne l'algorithme d'analyse en composantes principales (ACP ou PCA) ?",
        options: [
            "Il crée de nouvelles caractéristiques en sélectionnant les variables les plus importantes de l'ensemble de données original",
            "Il projette les données dans un espace de dimension inférieure en maximisant la variance conservée",
            "Il génère de nouvelles données synthétiques qui conservent les mêmes propriétés statistiques que les données originales",
            "Il regroupe les caractéristiques similaires pour réduire la redondance dans les données"
        ],
        correctIndex: 1,
        explanation: "L'analyse en composantes principales (ACP) est une technique de réduction de dimensionnalité non supervisée qui projette les données dans un espace de dimension inférieure tout en maximisant la variance conservée. L'algorithme fonctionne en : (1) centrant les données (et souvent en les standardisant) ; (2) calculant la matrice de covariance des caractéristiques ; (3) décomposant cette matrice en vecteurs propres et valeurs propres ; (4) sélectionnant les k vecteurs propres avec les plus grandes valeurs propres (correspondant aux directions de plus grande variance) ; (5) projetant les données originales sur ces k vecteurs propres. Les nouvelles caractéristiques (composantes principales) sont des combinaisons linéaires des caractéristiques originales, orthogonales entre elles, et ordonnées par la quantité de variance qu'elles expliquent. PCA est particulièrement utile pour visualiser des données de grande dimension, réduire la multicolinéarité, et compresser les données en préservant l'information la plus importante.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que l'apprentissage par transfert (transfer learning) et pourquoi est-il utile ?",
        options: [
            "Une méthode pour transférer un modèle d'un serveur à un autre, utile pour le déploiement distribué",
            "Une technique où l'on entraîne un modèle sur une tâche source puis on réutilise ses connaissances sur une tâche cible, utile quand les données de la tâche cible sont limitées",
            "Un algorithme qui transfère les étiquettes d'un jeu de données étiqueté à un jeu non étiqueté, utile pour l'apprentissage semi-supervisé",
            "Une approche qui convertit un modèle d'un framework à un autre, utile pour l'interopérabilité"
        ],
        correctIndex: 1,
        explanation: "L'apprentissage par transfert (transfer learning) est une technique où l'on entraîne d'abord un modèle sur une tâche source (généralement avec beaucoup de données) puis on réutilise une partie de ce modèle (typiquement les premières couches qui ont appris des représentations générales) pour une tâche cible différente mais liée. Cette approche est particulièrement utile lorsque les données pour la tâche cible sont limitées, car elle permet de s'appuyer sur les connaissances acquises lors de la tâche source, plutôt que de partir de zéro. En vision par ordinateur, par exemple, on utilise couramment des réseaux pré-entraînés sur ImageNet comme extracteurs de caractéristiques, puis on ajuste (fine-tune) les dernières couches pour des tâches spécifiques comme la détection d'objets particuliers. Cette approche réduit considérablement le temps d'entraînement, la quantité de données nécessaires et améliore souvent les performances, en particulier pour les problèmes où les données étiquetées sont rares ou coûteuses à obtenir.",
        difficulty: "medium"
    },
    {
        question: "Quelle est la différence entre bagging et boosting en apprentissage automatique ?",
        options: [
            "Le bagging améliore la précision, tandis que le boosting réduit la variance",
            "Le bagging entraîne les modèles séquentiellement, tandis que le boosting les entraîne en parallèle",
            "Le bagging pondère les prédictions finales, tandis que le boosting donne un poids égal à tous les modèles",
            "Le bagging entraîne des modèles indépendants en parallèle sur des sous-ensembles aléatoires, tandis que le boosting entraîne des modèles séquentiellement en se concentrant sur les erreurs précédentes"
        ],
        correctIndex: 3,
        explanation: "Le bagging (Bootstrap Aggregating) et le boosting sont deux approches d'ensemble fondamentalement différentes. Le bagging entraîne plusieurs modèles indépendamment en parallèle, chacun sur un sous-ensemble aléatoire des données obtenu par échantillonnage avec remplacement (bootstrap). Les prédictions finales sont généralement obtenues par vote majoritaire (classification) ou moyenne (régression). Cette approche réduit principalement la variance et aide à éviter le surapprentissage. Random Forest est un exemple d'algorithme utilisant le bagging. Le boosting, en revanche, entraîne les modèles séquentiellement, chaque nouveau modèle se concentrant davantage sur les exemples mal classifiés par les modèles précédents. Les modèles ne sont pas indépendants mais complémentaires, et le modèle final est une combinaison pondérée où les modèles plus performants ont plus de poids. Le boosting réduit principalement le biais et peut parfois surpasser la simple réduction de variance du bagging. AdaBoost, Gradient Boosting et XGBoost sont des exemples d'algorithmes de boosting.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que la régularisation L1 (Lasso) et comment diffère-t-elle de la régularisation L2 (Ridge) ?",
        options: [
            "La régularisation L1 ajoute le carré des paramètres à la fonction de coût, tandis que L2 ajoute leur valeur absolue",
            "La régularisation L1 ajoute la valeur absolue des paramètres à la fonction de coût et tend à produire des modèles parcimonieux, tandis que L2 ajoute le carré des paramètres et rétrécit uniformément tous les coefficients",
            "La régularisation L1 s'applique uniquement aux biais, tandis que L2 s'applique à tous les paramètres",
            "La régularisation L1 est utilisée pour la classification, tandis que L2 est utilisée pour la régression"
        ],
        correctIndex: 1,
        explanation: "La régularisation L1 (Lasso) ajoute à la fonction de coût un terme proportionnel à la somme des valeurs absolues des paramètres (||w||₁), tandis que la régularisation L2 (Ridge) ajoute un terme proportionnel à la somme des carrés des paramètres (||w||₂²). La principale différence est que L1 tend à produire des modèles parcimonieux, en mettant exactement à zéro certains coefficients, réalisant ainsi une sélection de caractéristiques implicite. Cela est dû à la géométrie de la pénalité L1, qui peut atteindre zéro sur certaines dimensions. En revanche, L2 rétrécit tous les coefficients de manière proportionnelle mais les maintient rarement exactement à zéro. L2 est particulièrement efficace pour gérer la multicolinéarité en distribuant le poids entre les caractéristiques corrélées. Dans la pratique, le choix entre L1 et L2 dépend de si l'on souhaite un modèle parcimonieux (L1) ou si l'on veut simplement contrôler la complexité sans éliminer complètement des caractéristiques (L2).",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que l'apprentissage semi-supervisé et quand est-il particulièrement utile ?",
        options: [
            "Une combinaison d'apprentissage supervisé et non supervisé où l'on utilise à la fois des données étiquetées et non étiquetées, utile quand l'étiquetage est coûteux",
            "Une approche où le modèle est partiellement supervisé par un expert humain pendant l'entraînement, utile pour les tâches complexes",
            "Une méthode qui alterne entre phases supervisées et non supervisées, utile pour les problèmes dynamiques",
            "Une technique qui utilise des étiquettes approximatives ou bruyantes, utile quand les étiquettes précises sont difficiles à obtenir"
        ],
        correctIndex: 0,
        explanation: "L'apprentissage semi-supervisé est une approche qui se situe entre l'apprentissage supervisé (qui utilise uniquement des données étiquetées) et l'apprentissage non supervisé (qui utilise uniquement des données non étiquetées). Il exploite à la fois une petite quantité de données étiquetées et une grande quantité de données non étiquetées. Cette approche est particulièrement utile dans les situations où l'obtention d'étiquettes est coûteuse, chronophage ou nécessite une expertise spécifique (comme en imagerie médicale), mais où les données non étiquetées sont abondantes. Les algorithmes semi-supervisés font généralement des hypothèses sur la structure des données, comme la continuité (points proches devraient avoir des étiquettes similaires) ou la structure en clusters (points dans le même cluster devraient partager la même étiquette). Des techniques comme la propagation d'étiquettes, les modèles génératifs, ou l'auto-apprentissage permettent d'utiliser les données non étiquetées pour améliorer les performances par rapport à un modèle purement supervisé utilisant seulement les données étiquetées disponibles.",
        difficulty: "medium"
    },
    {
        question: "Comment fonctionne l'algorithme DBSCAN (Density-Based Spatial Clustering of Applications with Noise) et en quoi diffère-t-il de k-means ?",
        options: [
            "DBSCAN regroupe les points en fonction de leur densité locale, ne nécessite pas de spécifier le nombre de clusters et peut détecter les outliers ; k-means minimise la distance intra-cluster, nécessite de spécifier k et suppose des clusters sphériques",
            "DBSCAN utilise la distance euclidienne, tandis que k-means peut utiliser n'importe quelle métrique de distance",
            "DBSCAN est déterministe, tandis que k-means est stochastique et dépend de l'initialisation",
            "DBSCAN est adapté aux petits jeux de données, tandis que k-means est plus efficace pour les grands volumes de données"
        ],
        correctIndex: 0,
        explanation: "DBSCAN et k-means diffèrent fondamentalement dans leur approche du clustering. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) regroupe les points en fonction de leur densité locale : il définit des points comme 'core points' s'ils ont un nombre minimum de voisins (minPts) dans un rayon spécifié (eps), puis construit des clusters en connectant les core points qui sont à portée l'un de l'autre. Les avantages de DBSCAN incluent : (1) il ne nécessite pas de spécifier à l'avance le nombre de clusters ; (2) il peut découvrir des clusters de forme arbitraire, pas seulement sphérique ; (3) il est robuste aux outliers, qui sont explicitement identifiés comme 'bruit'. En revanche, k-means vise à minimiser la somme des distances au carré entre les points et le centroïde de leur cluster. Il nécessite de spécifier le nombre k de clusters à l'avance, suppose que les clusters sont convexes et de taille/densité similaire, et est sensible aux outliers. K-means est généralement plus rapide et facile à implémenter, mais DBSCAN est plus flexible pour les données avec des distributions complexes.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que l'attaque par adversaires (adversarial attack) dans le contexte des réseaux de neurones profonds ?",
        options: [
            "Une technique d'entraînement où deux réseaux de neurones s'affrontent pour améliorer leurs performances",
            "Une méthode pour concevoir des réseaux de neurones plus robustes en ajoutant du bruit aux données d'entraînement",
            "Une tentative malveillante d'exploiter les faiblesses des modèles de machine learning en créant des exemples spécialement conçus pour tromper le modèle",
            "Une stratégie d'optimisation qui utilise deux fonctions objectif opposées pour trouver un équilibre optimal"
        ],
        correctIndex: 2,
        explanation: "Une attaque par adversaires (adversarial attack) est une technique qui vise à tromper un modèle de machine learning, en particulier les réseaux de neurones profonds, en créant des exemples d'entrée spécialement conçus (exemples adversariaux). Ces exemples contiennent des perturbations subtiles, souvent imperceptibles pour l'humain, mais qui causent le modèle à faire des erreurs de classification avec une grande confiance. Par exemple, en ajoutant un bruit calculé minutieusement à l'image d'un panda, on peut faire en sorte qu'un réseau de neurones la classifie comme un gibbon. Ces attaques exploitent le fait que les modèles de deep learning, malgré leur haute précision, peuvent être très sensibles à certaines directions dans l'espace des caractéristiques. Les attaques par adversaires soulèvent d'importantes préoccupations concernant la sécurité et la fiabilité des systèmes d'IA, particulièrement dans des applications critiques comme les véhicules autonomes ou la reconnaissance faciale. Des techniques comme l'entraînement adversarial ou la distillation défensive visent à rendre les modèles plus robustes face à ces attaques.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que l'apprentissage par renforcement et comment diffère-t-il de l'apprentissage supervisé ?",
        options: [
            "L'apprentissage par renforcement utilise des données étiquetées pour apprendre une fonction de mappage, tandis que l'apprentissage supervisé apprend par essai-erreur",
            "L'apprentissage par renforcement apprend par essai-erreur à travers des interactions avec un environnement, en maximisant une récompense, tandis que l'apprentissage supervisé apprend à partir d'exemples étiquetés",
            "L'apprentissage par renforcement est utilisé pour les tâches de régression, tandis que l'apprentissage supervisé est utilisé pour la classification",
            "L'apprentissage par renforcement nécessite moins de données que l'apprentissage supervisé"
        ],
        correctIndex: 1,
        explanation: "L'apprentissage par renforcement (RL) et l'apprentissage supervisé diffèrent fondamentalement dans leur approche. En apprentissage supervisé, le modèle apprend à partir d'un ensemble de données étiquetées, avec des entrées et sorties correctes explicites. L'objectif est de généraliser à partir de ces exemples pour faire des prédictions sur de nouvelles données. En revanche, l'apprentissage par renforcement implique un agent qui apprend à prendre des décisions en interagissant avec un environnement, sans exemples explicites de comportement optimal. L'agent reçoit des récompenses ou des pénalités en fonction de ses actions et apprend par essai-erreur à maximiser la récompense cumulée sur le long terme. Le RL est particulièrement adapté aux problèmes séquentiels où les actions actuelles influencent les états futurs et où le feedback peut être retardé (comme les jeux, la robotique ou le contrôle). Contrairement à l'apprentissage supervisé, le RL doit gérer le dilemme exploration-exploitation : explorer de nouvelles actions pour découvrir de meilleures stratégies ou exploiter les connaissances actuelles pour maximiser la récompense immédiate.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que l'embedding (plongement) de mots en traitement du langage naturel ?",
        options: [
            "Une technique de compression qui réduit la taille des documents textuels",
            "Un algorithme qui transforme les textes en graphiques pour faciliter la visualisation",
            "Une représentation vectorielle dense des mots qui capture leurs relations sémantiques et syntaxiques",
            "Une méthode pour extraire les mots-clés les plus importants d'un texte"
        ],
        correctIndex: 2,
        explanation: "L'embedding (ou plongement) de mots est une technique de représentation vectorielle dense des mots dans un espace continu de dimension relativement faible (typiquement 50 à 300 dimensions). Contrairement aux représentations one-hot qui sont éparses et de très haute dimension, les embeddings capturent les relations sémantiques et syntaxiques entre les mots : des mots similaires ou liés sont proches dans cet espace vectoriel. Cette propriété permet des opérations algébriques significatives, comme dans l'exemple classique 'roi - homme + femme ≈ reine'. Les embeddings sont généralement appris à partir de grands corpus de texte en utilisant des modèles comme Word2Vec, GloVe ou FastText, qui exploitent le principe que les mots apparaissant dans des contextes similaires ont tendance à avoir des significations similaires. Ces représentations vectorielles sont devenues un composant fondamental en NLP, servant d'entrée à des modèles plus complexes pour diverses tâches comme la classification de texte, la traduction automatique, ou les systèmes de question-réponse.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que l'architecture Transformer et pourquoi a-t-elle révolutionné le traitement du langage naturel ?",
        options: [
            "Une architecture qui transforme les mots en vecteurs en utilisant des réseaux de neurones convolutifs",
            "Une architecture basée entièrement sur des mécanismes d'attention qui a surpassé les RNN en capturant efficacement les dépendances à long terme sans traitement séquentiel",
            "Un modèle qui transforme les données textuelles en représentations visuelles pour faciliter l'analyse",
            "Une architecture hybride qui combine CNN et RNN pour optimiser la performance dans les tâches de NLP"
        ],
        correctIndex: 1,
        explanation: "L'architecture Transformer, introduite dans l'article \"Attention Is All You Need\" (2017), a révolutionné le traitement du langage naturel en remplaçant les architectures récurrentes (RNN, LSTM) par un modèle basé entièrement sur des mécanismes d'attention. Sa principale innovation est l'attention multi-têtes (multi-head attention) qui permet au modèle de se concentrer simultanément sur différentes parties d'une séquence, capturant efficacement les dépendances à long terme sans nécessiter de traitement séquentiel. Cette parallélisation rend les Transformers beaucoup plus rapides à entraîner que les RNN. De plus, l'architecture encodeur-décodeur et l'attention masquée permettent d'excellentes performances sur diverses tâches comme la traduction, la génération de texte, et la compréhension du langage. Les Transformers ont conduit à des modèles pré-entraînés puissants comme BERT, GPT, et T5, qui ont établi de nouveaux standards d'état de l'art dans presque toutes les tâches de NLP, démontrant une capacité remarquable à capturer les nuances linguistiques et à transférer les connaissances à travers différentes tâches.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que le compromis biais-variance en apprentissage automatique ?",
        options: [
            "Un compromis entre le temps d'entraînement et la précision du modèle",
            "Un compromis entre le nombre de faux positifs et de faux négatifs dans un modèle de classification",
            "Un compromis entre sous-apprentissage (high bias, modèle trop simple) et surapprentissage (high variance, modèle trop complexe)",
            "Un compromis entre la quantité de données nécessaires et la complexité du modèle"
        ],
        correctIndex: 2,
        explanation: "Le compromis biais-variance est un concept fondamental en apprentissage automatique qui décrit la tension entre deux sources d'erreur dans les modèles prédictifs. Le biais (bias) représente l'erreur due à des hypothèses simplificatrices dans le modèle. Un modèle à haut biais (comme une régression linéaire simple face à des données non linéaires) tend à sous-apprendre : il ne capture pas la complexité sous-jacente des données et performe mal même sur les données d'entraînement. La variance, en revanche, représente la sensibilité du modèle aux fluctuations dans les données d'entraînement. Un modèle à haute variance (comme un arbre de décision très profond) tend à surpprendre : il capture le bruit dans les données d'entraînement plutôt que le signal sous-jacent, performant bien sur ces données mais se généralisant mal à de nouvelles données. L'art de la modélisation consiste à trouver le juste équilibre entre ces deux extrêmes. Des techniques comme la régularisation, la validation croisée, et les ensembles aident à gérer ce compromis pour obtenir des modèles qui généralisent bien.",
        difficulty: "medium"
    },
    {
        question: "Comment fonctionne l'algorithme des forêts aléatoires (Random Forest) et quels sont ses avantages ?",
        options: [
            "Il construit plusieurs arbres de décision indépendants en introduisant de l'aléatoire via le bootstrap et la sélection aléatoire de caractéristiques, offrant une réduction de variance, une robustesse au surapprentissage et une bonne gestion des outliers",
            "Il sélectionne aléatoirement un sous-ensemble de caractéristiques pour construire un arbre de décision optimal",
            "Il intègre des composantes aléatoires dans un seul arbre de décision pour améliorer sa robustesse",
            "Il combine des arbres de décision avec d'autres types de modèles choisis aléatoirement pour diversifier les prédictions"
        ],
        correctIndex: 0,
        explanation: "L'algorithme Random Forest (forêts aléatoires) est une méthode d'ensemble qui construit de nombreux arbres de décision indépendants et combine leurs prédictions. L'aléatoire est introduit de deux façons : (1) chaque arbre est entraîné sur un échantillon bootstrap des données (tirage avec remise), et (2) à chaque division, seul un sous-ensemble aléatoire des caractéristiques est considéré. Ces sources d'aléatoire garantissent que les arbres sont diversifiés, ce qui réduit la variance sans augmenter le biais. Les avantages des forêts aléatoires incluent : une excellente précision sur divers types de problèmes ; une robustesse au surapprentissage (particulièrement avec de nombreux arbres) ; une bonne gestion des données de haute dimension et des caractéristiques non pertinentes ; la capacité à gérer des valeurs manquantes et des outliers ; et une mesure d'importance des caractéristiques intégrée. De plus, l'algorithme est facile à paramétrer et naturellement parallélisable. Ces propriétés en font une des méthodes d'apprentissage automatique les plus populaires et polyvalentes.",
        difficulty: "medium"
    },
    {
        question: "Qu'est-ce que le Q-learning dans le contexte de l'apprentissage par renforcement ?",
        options: [
            "Un algorithme qui optimise directement la politique de l'agent via des méthodes de gradient de politique",
            "Une méthode pour quantifier (d'où le 'Q') la qualité des données d'entraînement",
            "Un algorithme d'apprentissage par renforcement sans modèle qui apprend une fonction de valeur d'action (Q-function) pour déterminer la politique optimale",
            "Une technique d'apprentissage supervisé spécialisée pour les problèmes avec retour d'information retardé"
        ],
        correctIndex: 2,
        explanation: "Le Q-learning est un algorithme fondamental d'apprentissage par renforcement sans modèle (model-free), qui apprend une fonction de valeur d'action appelée Q-function. Cette fonction Q(s,a) représente la récompense totale espérée en prenant l'action a dans l'état s puis en suivant une politique optimale. L'algorithme mise à jour itérativement cette fonction en utilisant l'équation de Bellman : Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)], où α est le taux d'apprentissage, γ le facteur d'actualisation, r la récompense immédiate, et s' l'état suivant. Une fois la Q-function apprise, la politique optimale consiste simplement à choisir l'action qui maximise Q dans chaque état. Le Q-learning est 'sans modèle' car il n'apprend pas explicitement les probabilités de transition entre les états, et 'off-policy' car il peut apprendre la politique optimale en suivant n'importe quelle politique d'exploration. Le Deep Q-Network (DQN) est une extension utilisant des réseaux de neurones profonds pour approximer la Q-function dans des espaces d'états complexes, comme dans les jeux vidéo.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que le pruning (élagage) dans les arbres de décision et pourquoi est-il utile ?",
        options: [
            "Une technique pour supprimer les branches peu importantes d'un arbre déjà construit, utile pour réduire le surapprentissage",
            "Une méthode pour limiter la profondeur maximale de l'arbre pendant sa construction",
            "Un algorithme qui supprime les caractéristiques redondantes avant de construire l'arbre",
            "Une stratégie pour fusionner des arbres similaires dans un ensemble comme Random Forest"
        ],
        correctIndex: 0,
        explanation: "Le pruning (élagage) dans les arbres de décision est une technique qui consiste à supprimer certaines branches d'un arbre déjà construit, généralement celles qui apportent peu d'information ou qui semblent capturer du bruit plutôt que du signal. Il existe deux approches principales : le pré-élagage (pre-pruning), qui arrête la croissance de l'arbre pendant sa construction en utilisant des critères comme la profondeur maximale ou le nombre minimum d'exemples par feuille, et le post-élagage (post-pruning), qui construit d'abord un arbre complet puis le simplifie en éliminant les branches qui ne contribuent pas significativement à la précision sur un ensemble de validation. L'élagage est principalement utile pour réduire le surapprentissage : un arbre trop complexe peut parfaitement s'ajuster aux données d'entraînement mais généraliser médiocrement aux nouvelles données. En réduisant la complexité du modèle, le pruning améliore sa capacité de généralisation, sa robustesse au bruit, et aussi son interprétabilité en produisant des arbres plus compacts et lisibles.",
        difficulty: "medium"
    },
    {
        question: "Quelle est la différence entre un modèle paramétrique et non paramétrique en apprentissage automatique ?",
        options: [
            "Un modèle paramétrique n'a pas de paramètres à apprendre, tandis qu'un modèle non paramétrique en a",
            "Un modèle paramétrique a un nombre fixe de paramètres indépendant de la taille des données, tandis qu'un modèle non paramétrique peut augmenter sa complexité avec plus de données",
            "Un modèle paramétrique est plus complexe et plus précis qu'un modèle non paramétrique",
            "Un modèle paramétrique nécessite une phase d'entraînement, contrairement à un modèle non paramétrique"
        ],
        correctIndex: 1,
        explanation: "La distinction fondamentale entre modèles paramétrique et non paramétrique réside dans leur relation avec les données. Un modèle paramétrique fait des hypothèses fortes sur la forme de la fonction sous-jacente et possède un nombre fixe de paramètres, indépendant de la taille des données d'entraînement. Des exemples incluent la régression linéaire, la régression logistique, ou les réseaux de neurones avec architecture fixe. Une fois les paramètres appris, les données d'entraînement peuvent être oubliées. En revanche, un modèle non paramétrique fait moins d'hypothèses préalables et peut adapter sa complexité en fonction de la quantité de données disponibles. Le nombre 'effectif' de paramètres peut augmenter avec la taille du jeu de données, permettant de capturer des relations plus complexes avec plus de données. Des exemples incluent les k plus proches voisins, les arbres de décision, ou les méthodes à noyau comme les SVM avec noyaux non linéaires. Ces modèles conservent généralement les données d'entraînement (ou une transformation de celles-ci) pour faire des prédictions. Cette flexibilité des modèles non paramétriques les rend souvent plus adaptés aux problèmes où la forme de la fonction sous-jacente est inconnue ou complexe.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que le théorème No Free Lunch en apprentissage automatique ?",
        options: [
            "Un principe stipulant que l'apprentissage automatique ne peut pas résoudre tous les problèmes, certains nécessitant toujours une expertise humaine",
            "Une règle empirique selon laquelle tout algorithme a besoin d'une quantité minimale de données pour bien fonctionner",
            "Un théorème mathématique démontrant qu'aucun algorithme d'apprentissage n'est universellement supérieur à tous les autres sur tous les problèmes possibles",
            "Une observation que pour chaque amélioration de performance, il y a toujours un coût computationnel proportionnel"
        ],
        correctIndex: 2,
        explanation: "Le théorème No Free Lunch (NFL), formalisé par Wolpert et Macready, est un résultat théorique fondamental en apprentissage automatique qui stipule qu'aucun algorithme d'apprentissage n'est universellement supérieur à tous les autres sur l'ensemble des problèmes possibles. En moyennant sur tous les problèmes mathématiquement possibles, tous les algorithmes ont des performances identiques. Cela implique qu'il n'existe pas d'algorithme 'miracle' qui serait optimal pour tous les cas : les performances supérieures d'un algorithme sur certains types de problèmes sont nécessairement compensées par des performances inférieures sur d'autres. Ce théorème justifie pourquoi la sélection d'algorithmes et l'ingénierie des caractéristiques restent essentielles en apprentissage automatique : il faut adapter l'approche aux spécificités du problème et incorporer des connaissances préalables pertinentes. En pratique, les problèmes du monde réel ne sont pas distribués uniformément dans l'espace des problèmes possibles, donc certains algorithmes tendent à mieux performer sur les types de problèmes que nous rencontrons fréquemment.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que l'inférence bayésienne et comment diffère-t-elle de l'approche fréquentiste en statistique ?",
        options: [
            "L'inférence bayésienne utilise uniquement des probabilités conditionnelles, tandis que l'approche fréquentiste utilise uniquement des probabilités marginales",
            "L'inférence bayésienne considère les paramètres comme des variables aléatoires avec des distributions de probabilité, incorporant des connaissances préalables via des priors, tandis que l'approche fréquentiste les traite comme des valeurs fixes inconnues",
            "L'inférence bayésienne s'applique uniquement aux problèmes de classification, tandis que l'approche fréquentiste s'applique à tous les types de problèmes statistiques",
            "L'inférence bayésienne utilise des méthodes de Monte Carlo, tandis que l'approche fréquentiste utilise des formules analytiques exactes"
        ],
        correctIndex: 1,
        explanation: "L'inférence bayésienne et l'approche fréquentiste représentent deux paradigmes fondamentalement différents en statistique. Dans l'approche bayésienne, les paramètres du modèle sont considérés comme des variables aléatoires décrites par des distributions de probabilité. Elle commence par spécifier une distribution a priori (prior) qui encode les croyances ou connaissances préalables sur les paramètres avant d'observer les données. Cette distribution est ensuite mise à jour en fonction des données observées via le théorème de Bayes, résultant en une distribution a posteriori qui représente les croyances mises à jour. L'approche fréquentiste, en revanche, traite les paramètres comme des valeurs fixes mais inconnues. Elle définit la probabilité comme la limite de la fréquence relative d'un événement lorsqu'une expérience est répétée indéfiniment. Les méthodes fréquentistes comme les tests d'hypothèses, les p-values et les intervalles de confiance sont basées sur cette interprétation. Les différences clés incluent : l'incorporation de connaissances préalables (bayésien) vs. l'objectivité apparente (fréquentiste), l'interprétation directe des probabilités sur les paramètres (bayésien) vs. les déclarations sur la fréquence à long terme des procédures statistiques (fréquentiste), et la possibilité de mise à jour incrémentale des croyances (bayésien) qui n'existe pas dans le cadre fréquentiste.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce qu'un réseau adversaire génératif (GAN) et comment fonctionne-t-il ?",
        options: [
            "Un réseau qui génère des exemples adversariaux pour tester la robustesse d'autres modèles",
            "Un système de deux réseaux de neurones (générateur et discriminateur) qui s'affrontent dans un jeu à somme nulle, le générateur créant des données synthétiques tandis que le discriminateur tente de distinguer données réelles et synthétiques",
            "Un algorithme qui combine plusieurs réseaux de neurones pour résoudre des problèmes adversariaux comme les jeux à deux joueurs",
            "Une technique pour entraîner un réseau de neurones à reconnaître et neutraliser les attaques adversariales"
        ],
        correctIndex: 1,
        explanation: "Un réseau adversaire génératif (GAN, Generative Adversarial Network) est un framework d'apprentissage non supervisé où deux réseaux de neurones s'affrontent dans un jeu à somme nulle. Le système comprend deux composantes principales : le générateur (G) et le discriminateur (D). Le générateur tente de produire des données synthétiques qui ressemblent aux données réelles, tandis que le discriminateur essaie de distinguer les échantillons réels des échantillons générés. Mathématiquement, ils jouent un jeu minimax : le générateur cherche à minimiser la capacité du discriminateur à faire la distinction, tandis que le discriminateur cherche à maximiser sa capacité de discrimination. Pendant l'entraînement, les deux réseaux s'améliorent progressivement : le générateur produit des échantillons de plus en plus réalistes, et le discriminateur devient de plus en plus précis dans sa capacité à détecter les faux. À l'équilibre théorique, les échantillons générés sont indiscernables des données réelles, et le discriminateur ne peut pas faire mieux que deviner au hasard (50% de précision). Les GANs ont révolutionné la génération d'images, permettant de créer des photos réalistes de personnes inexistantes, de convertir des croquis en images photoréalistes, ou de transférer des styles artistiques.",
        difficulty: "hard"
    },
    {
        question: "Qu'est-ce que le curriculum learning (apprentissage par curriculum) en deep learning ?",
        options: [
            "Une technique où l'on commence l'entraînement avec des données simples puis on augmente progressivement la difficulté, imitant l'apprentissage humain",
            "Une méthode qui alterne entre différents types de tâches d'apprentissage pour améliorer la généralisation",
            "Un processus d'apprentissage continu où le modèle est régulièrement mis à jour avec de nouvelles données",
            "Une approche qui divise un problème complexe en sous-problèmes plus simples à résoudre séquentiellement"
        ],
        correctIndex: 0,
        explanation: "Le curriculum learning (apprentissage par curriculum) est une stratégie d'entraînement inspirée de la façon dont les humains apprennent, en commençant par des concepts simples avant de passer à des concepts plus complexes. Dans ce paradigme, le modèle est d'abord exposé à des exemples faciles à apprendre, puis progressivement à des exemples de plus en plus difficiles. Cette approche contraste avec l'entraînement standard où tous les exemples sont présentés de manière aléatoire, indépendamment de leur difficulté. L'idée sous-jacente est que commencer par des exemples simples établit une base solide qui facilite ensuite l'apprentissage des cas plus complexes, potentiellement en guidant l'optimisation vers de meilleurs minima locaux. La difficulté peut être définie de diverses manières selon la tâche : simplicité des structures grammaticales en traitement du langage, clarté des images en vision par ordinateur, ou complexité des environnements en apprentissage par renforcement. Des études empiriques ont montré que cette approche peut accélérer la convergence et parfois améliorer les performances finales, particulièrement pour les problèmes complexes ou lorsque les données présentent une grande variabilité.",
        difficulty: "medium"
    }
];

// Exporter les questions
if (typeof module !== 'undefined') {
    module.exports = { questionsTheoriques };
}