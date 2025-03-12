// questions-theoriques.js
// Questions théoriques sur l'IA

const questionsTheoriques = [
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
];

// Exporter les questions
if (typeof module !== 'undefined') {
    module.exports = { questionsTheoriques };
}