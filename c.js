// questions-code.js 
// 30 Questions code Python pour l'IA

const questionsCode = [
    // Question 1
    {
        question: "Comment compléter ce code pour calculer le biais et la variance dans la décomposition de l'erreur de classification ?\ndef bias_variance_decomp_classification(model, X, y, n_splits=5):\n    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n    predictions = []\n    actual = []\n    \n    # Collecte des prédictions pour chaque fold\n    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):\n        X_train_fold, X_val_fold = X[train_idx], X[val_idx]\n        y_train_fold, y_val_fold = y[train_idx], y[val_idx]\n        ...",
        options: [
            "        model.fit(X_train_fold, y_train_fold)\n        # Prédictions sur l'ensemble de validation\n        pred_proba = model.predict_proba(X_val_fold)\n        predictions.append(pred_proba)\n        actual.append(y_val_fold)\n\n    # Convertir en arrays numpy pour faciliter les calculs\n    predictions = np.array(predictions)\n    actual = np.array(actual)\n\n    # Calcul du biais\n    avg_pred = np.mean(predictions, axis=0)\n    bias = np.mean((avg_pred.argmax(axis=1) - actual.flatten()) ** 2)\n\n    # Calcul de la variance\n    variance = np.mean(np.var(predictions, axis=0))\n\n    return bias, variance",
            "        model.fit(X_train_fold, y_train_fold)\n        # Prédictions sur l'ensemble de validation\n        pred = model.predict(X_val_fold)\n        predictions.append(pred)\n        actual.append(y_val_fold)\n\n    # Convertir en arrays numpy\n    predictions = np.array(predictions)\n    actual = np.array(actual)\n\n    # Calcul du biais\n    mean_pred = np.mean(predictions, axis=0)\n    bias = np.mean((mean_pred - actual.flatten()) ** 2)\n\n    # Calcul de la variance\n    var = np.zeros(len(predictions[0]))\n    for i in range(len(predictions)):\n        var += (predictions[i] - mean_pred) ** 2\n    variance = np.mean(var / len(predictions))\n\n    return bias, variance",
            "        model.fit(X_train_fold, y_train_fold)\n        # Prédictions sur l'ensemble de validation\n        pred = model.predict(X_val_fold)\n        predictions.append(pred)\n        actual.extend(y_val_fold)\n\n    # Convertir en arrays\n    predictions = np.array(predictions)\n    actual = np.array(actual)\n\n    # Calcul du biais et de la variance\n    ensemble_pred = np.mean(predictions, axis=0)\n    bias = np.mean((ensemble_pred - actual) ** 2)\n    variance = np.mean([np.mean((pred - ensemble_pred) ** 2) for pred in predictions])\n\n    return bias, variance",
            "        model.fit(X_train_fold, y_train_fold)\n        pred = model.predict_proba(X_val_fold)[:, 1]  # Prob. classe positive\n        predictions.append(pred)\n        actual.append(y_val_fold)\n\n    # Convertir en arrays numpy\n    predictions = np.stack(predictions, axis=0)\n    actual = np.concatenate(actual)\n\n    # Moyenne des prédictions de tous les modèles pour chaque instance\n    avg_pred = np.mean(predictions, axis=0)\n\n    # Biais - erreur quadratique entre prédiction moyenne et vérité\n    bias = np.mean((avg_pred - actual) ** 2)\n\n    # Variance - variance moyenne des prédictions pour chaque instance\n    variance = np.mean(np.var(predictions, axis=0))\n\n    return bias, variance"
        ],
        correctIndex: 3,
        explanation: "Cette solution est correcte car elle : (1) Utilise predict_proba pour obtenir les probabilités de la classe positive, essentiel pour la décomposition biais-variance en classification, (2) Empile correctement les prédictions avec np.stack pour garder la structure des folds, (3) Calcule le biais comme l'erreur quadratique moyenne entre la prédiction moyenne de tous les modèles et les valeurs réelles, (4) Calcule la variance comme la variance moyenne des prédictions entre les différents modèles pour chaque instance, capturant ainsi la variabilité due à la sensibilité aux données d'entraînement.",
        difficulty: "hard"
    },

    // Question 2
    {
        question: "Comment implémenter la validation croisée pour un modèle XGBoost avec early stopping en Python ?",
        options: [
            "from sklearn.model_selection import cross_val_score\nfrom xgboost import XGBClassifier\n\n# Création du modèle\nmodel = XGBClassifier(n_estimators=1000, early_stopping_rounds=10)\n\n# Validation croisée\nscores = cross_val_score(model, X, y, cv=5)",
            "from sklearn.model_selection import KFold\nimport xgboost as xgb\nimport numpy as np\n\ndef xgb_cv_with_early_stopping(X, y, params, num_folds=5, early_stopping_rounds=50):\n    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)\n    cv_scores = []\n    \n    for train_idx, val_idx in kf.split(X):\n        X_train, X_val = X[train_idx], X[val_idx]\n        y_train, y_val = y[train_idx], y[val_idx]\n        \n        # Créer les datasets XGBoost\n        dtrain = xgb.DMatrix(X_train, label=y_train)\n        dval = xgb.DMatrix(X_val, label=y_val)\n        \n        # Entraîner avec early stopping\n        model = xgb.train(\n            params,\n            dtrain,\n            num_boost_round=1000,\n            evals=[(dtrain, 'train'), (dval, 'val')],\n            early_stopping_rounds=early_stopping_rounds,\n            verbose_eval=False\n        )\n        \n        # Évaluer le modèle sur l'ensemble de validation\n        pred = model.predict(dval)\n        score = compute_metric(y_val, pred)  # Fonction à définir selon la métrique souhaitée\n        cv_scores.append(score)\n    \n    return np.mean(cv_scores), np.std(cv_scores)",
            "from sklearn.model_selection import KFold\nfrom xgboost import XGBClassifier\n\ndef custom_cv_xgb(X, y, n_splits=5):\n    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)\n    scores = []\n    best_iterations = []\n    \n    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):\n        X_train, X_val = X[train_idx], X[val_idx]\n        y_train, y_val = y[train_idx], y[val_idx]\n        \n        # Créer et entraîner le modèle avec early stopping\n        model = XGBClassifier(\n            n_estimators=1000,\n            learning_rate=0.1,\n            max_depth=6,\n            objective='binary:logistic'\n        )\n        \n        # Utiliser eval_set pour early stopping\n        model.fit(\n            X_train, y_train,\n            eval_set=[(X_val, y_val)],\n            early_stopping_rounds=20,\n            eval_metric='logloss',\n            verbose=False\n        )\n        \n        # Récupérer le nombre optimal d'itérations\n        best_iteration = model.best_iteration\n        best_iterations.append(best_iteration)\n        \n        # Scorer le modèle\n        score = model.score(X_val, y_val)\n        scores.append(score)\n        \n    # Retourner les résultats de la validation croisée\n    return np.mean(scores), np.std(scores), np.mean(best_iterations)",
            "from sklearn.model_selection import StratifiedKFold\nimport xgboost as xgb\nimport numpy as np\n\ndef nested_cv_xgboost(X, y, param_grid, n_outer=5, n_inner=3):\n    # Boucle externe pour estimer la performance du modèle final\n    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)\n    outer_scores = []\n    \n    for outer_train_idx, test_idx in outer_cv.split(X, y):\n        X_outer_train, X_test = X[outer_train_idx], X[test_idx]\n        y_outer_train, y_test = y[outer_train_idx], y[test_idx]\n        \n        # Boucle interne pour la sélection d'hyperparamètres\n        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=43)\n        best_score = -np.inf\n        best_params = None\n        \n        for params in param_grid:\n            inner_scores = []\n            \n            for inner_train_idx, val_idx in inner_cv.split(X_outer_train, y_outer_train):\n                X_inner_train, X_val = X_outer_train[inner_train_idx], X_outer_train[val_idx]\n                y_inner_train, y_val = y_outer_train[inner_train_idx], y_outer_train[val_idx]\n                \n                # Convertir en DMatrix pour XGBoost\n                dtrain = xgb.DMatrix(X_inner_train, label=y_inner_train)\n                dval = xgb.DMatrix(X_val, label=y_val)\n                \n                # Entraîner avec early stopping\n                xgb_model = xgb.train(\n                    params,\n                    dtrain,\n                    num_boost_round=1000,\n                    evals=[(dval, 'val')],\n                    early_stopping_rounds=50,\n                    verbose_eval=False\n                )\n                \n                # Évaluer\n                val_pred = xgb_model.predict(dval)\n                score = calculate_metric(y_val, val_pred)  # Fonction à définir\n                inner_scores.append(score)\n            \n            # Moyenne des scores sur les plis internes\n            mean_score = np.mean(inner_scores)\n            if mean_score > best_score:\n                best_score = mean_score\n                best_params = params\n        \n        # Entraîner le modèle final avec les meilleurs paramètres\n        dtrain_full = xgb.DMatrix(X_outer_train, label=y_outer_train)\n        dtest = xgb.DMatrix(X_test, label=y_test)\n        \n        final_model = xgb.train(\n            best_params,\n            dtrain_full,\n            num_boost_round=1000,\n            evals=[(dtest, 'test')],\n            early_stopping_rounds=50,\n            verbose_eval=False\n        )\n        \n        # Évaluer le modèle final\n        test_pred = final_model.predict(dtest)\n        test_score = calculate_metric(y_test, test_pred)\n        outer_scores.append(test_score)\n    \n    # Retourner la performance moyenne sur les plis externes\n    return np.mean(outer_scores), np.std(outer_scores)"
        ],
        correctIndex: 2,
        explanation: "La méthode custom_cv_xgb est la plus appropriée pour implémenter la validation croisée avec early stopping pour XGBoost. En utilisant l'API scikit-learn de XGBoost (XGBClassifier), on peut facilement intégrer l'early stopping grâce au paramètre eval_set qui permet de spécifier un ensemble de validation, et early_stopping_rounds qui arrête l'entraînement si la métrique d'évaluation ne s'améliore pas pendant un certain nombre d'itérations. Cette approche est plus simple et directe que l'utilisation de l'API native de XGBoost (qui nécessite la création de DMatrix), tout en permettant de récupérer le nombre optimal d'itérations pour chaque fold.",
        difficulty: "hard"
    },

    // Question 3
    {
        question: "Comment implémenter en Python une fonction pour évaluer l'importance des caractéristiques dans un modèle Random Forest basée sur la permutation ?",
        options: [
            "def permutation_importance(model, X, y, n_repeats=10):\n    # Score de base\n    baseline_score = model.score(X, y)\n    \n    # Calculer l'importance par permutation\n    n_features = X.shape[1]\n    importances = np.zeros(n_features)\n    importances_std = np.zeros(n_features)\n    \n    for i in range(n_features):\n        feature_importances = []\n        \n        for _ in range(n_repeats):\n            # Copier les données\n            X_permuted = X.copy()\n            \n            # Permuter la caractéristique i\n            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])\n            \n            # Calculer la baisse de performance\n            permuted_score = model.score(X_permuted, y)\n            feature_importance = baseline_score - permuted_score\n            feature_importances.append(feature_importance)\n        \n        # Moyenne et écart-type sur les répétitions\n        importances[i] = np.mean(feature_importances)\n        importances_std[i] = np.std(feature_importances)\n    \n    return importances, importances_std",
            "def feature_importance(model, X):\n    # Récupérer l'importance des caractéristiques depuis le modèle\n    importances = model.feature_importances_\n    \n    # Calculer l'écart-type entre les arbres\n    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)\n    \n    # Créer un DataFrame pour les résultats\n    indices = np.argsort(importances)[::-1]\n    feature_names = [f'feature_{i}' for i in range(X.shape[1])]\n    \n    result = pd.DataFrame({\n        'feature': [feature_names[i] for i in indices],\n        'importance': importances[indices],\n        'std': std[indices]\n    })\n    \n    return result",
            "from sklearn.inspection import permutation_importance\n\ndef compute_feature_importance(model, X, y):\n    # Calculer l'importance des caractéristiques basée sur la permutation\n    result = permutation_importance(\n        model, X, y, n_repeats=10, random_state=42\n    )\n    \n    # Trier les caractéristiques par importance\n    feature_names = [f'feature_{i}' for i in range(X.shape[1])]\n    importances = pd.DataFrame({\n        'feature': feature_names,\n        'importance': result.importances_mean,\n        'std': result.importances_std\n    })\n    \n    return importances.sort_values('importance', ascending=False)",
            "def drop_column_importance(model, X, y):\n    # Score de base\n    baseline_score = model.score(X, y)\n    \n    # Calculer l'importance en supprimant chaque caractéristique\n    n_features = X.shape[1]\n    importances = np.zeros(n_features)\n    \n    for i in range(n_features):\n        # Créer un jeu de données sans la caractéristique i\n        X_dropped = np.delete(X, i, axis=1)\n        \n        # Ré-entraîner le modèle\n        model_dropped = clone(model)\n        model_dropped.fit(X_dropped, y)\n        \n        # Calculer la baisse de performance\n        dropped_score = model_dropped.score(X_dropped, y)\n        importances[i] = baseline_score - dropped_score\n    \n    return importances"
        ],
        correctIndex: 0,
        explanation: "La première option implémente correctement l'importance par permutation, qui est une méthode model-agnostic pour évaluer l'importance des caractéristiques. L'idée fondamentale est de mesurer la baisse de performance lorsqu'une caractéristique spécifique est 'cassée' (rendue aléatoire par permutation), tout en laissant les autres intactes. Si la performance chute significativement, cela signifie que la caractéristique est importante. Contrairement à l'attribut feature_importances_ natif de RandomForest (qui est basé sur la diminution de l'impureté), la méthode par permutation peut capturer des interactions complexes et fonctionne même après l'entraînement du modèle. La répétition multiple (n_repeats) permet d'obtenir une estimation plus stable et de quantifier l'incertitude via l'écart-type.",
        difficulty: "hard"
    },

    // Question 4
    {
        question: "Parmi les implémentations suivantes pour la normalisation des caractéristiques par lot (batch normalization), laquelle est correcte ?",
        options: [
            "def batch_normalization(X, epsilon=1e-8):\n    # Normaliser chaque caractéristique\n    mean = np.mean(X, axis=0)\n    std = np.std(X, axis=0)\n    return (X - mean) / (std + epsilon)",
            "import tensorflow as tf\n\ndef add_batch_norm(model, inputs):\n    return tf.keras.layers.BatchNormalization()(inputs)",
            "class BatchNormalization:\n    def __init__(self, momentum=0.9, epsilon=1e-5):\n        self.momentum = momentum\n        self.epsilon = epsilon\n        self.running_mean = None\n        self.running_var = None\n        self.gamma = None  # Scale parameter\n        self.beta = None   # Shift parameter\n        \n    def forward(self, X, training=True):\n        # Initialize parameters if not done yet\n        if self.gamma is None:\n            self.gamma = np.ones(X.shape[1])\n            self.beta = np.zeros(X.shape[1])\n            self.running_mean = np.zeros(X.shape[1])\n            self.running_var = np.ones(X.shape[1])\n        \n        if training:\n            # Compute batch statistics\n            batch_mean = np.mean(X, axis=0)\n            batch_var = np.var(X, axis=0)\n            \n            # Update running statistics\n            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean\n            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var\n            \n            # Normalize\n            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)\n        else:\n            # Use running statistics for inference\n            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)\n        \n        # Scale and shift\n        return self.gamma * X_norm + self.beta",
            "from sklearn.preprocessing import StandardScaler\n\ndef normalize_features(X_train, X_test):\n    scaler = StandardScaler()\n    X_train_scaled = scaler.fit_transform(X_train)\n    X_test_scaled = scaler.transform(X_test)\n    return X_train_scaled, X_test_scaled"
        ],
        correctIndex: 2,
        explanation: "La troisième option implémente correctement la batch normalization telle que décrite dans l'article original de Ioffe et Szegedy (2015). Les éléments clés sont : (1) Calcul des statistiques par lot pendant l'entraînement (moyenne et variance de chaque mini-lot), (2) Maintien de statistiques glissantes (running_mean, running_var) pour l'inférence, contrôlées par le paramètre momentum, (3) Normalisation avec un terme epsilon pour éviter la division par zéro, (4) Paramètres apprenables gamma (échelle) et beta (décalage) qui permettent au réseau d'annuler la normalisation si nécessaire. Les autres options manquent certains aspects essentiels : la première est une simple normalisation Z-score, la seconde est juste un appel à l'API TensorFlow sans détails d'implémentation, et la quatrième est une standardisation globale avec scikit-learn, pas une normalisation par lot.",
        difficulty: "hard"
    },

    // Question 5
    {
        question: "Comment implémenter un modèle k-NN pour la classification en utilisant scikit-learn ?",
        options: [
          "from sklearn.neighbors import KNeighborsClassifier\n\ndef train_knn(X_train, y_train, k=5):\n    knn = KNeighborsClassifier(n_neighbors=k)\n    knn.fit(X_train, y_train)\n    return knn",
          "from sklearn.ensemble import KNNClassifier\n\ndef train_knn(X_train, y_train, k=5):\n    knn = KNNClassifier(neighbors=k)\n    knn.train(X_train, y_train)\n    return knn",
          "import numpy as np\n\ndef knn_predict(X_train, y_train, X_test, k=5):\n    # Code incomplet ou erroné",
          "from sklearn.metrics import pairwise_distances\n\ndef knn_classifier(X_train, y_train, X_test, k=5):\n    distances = pairwise_distances(X_test, X_train)\n    nearest = np.argsort(distances, axis=1)[:, :k]\n    predictions = [np.argmax(np.bincount(y_train[neighbors])) for neighbors in nearest]\n    return predictions"
        ],
        correctIndex: 0,
        explanation: "La première option est la bonne implémentation du k‑NN en utilisant scikit-learn, car elle utilise KNeighborsClassifier pour créer et entraîner le modèle de manière simple et efficace.",
        difficulty: "easy"
      }
      ,

    // Question 6
    {
        question: "Comment implémenter une régression logistique avec régularisation L1 (Lasso) en scikit-learn ?",
        options: [
            "from sklearn.linear_model import LogisticRegressionL1\n\nmodel = LogisticRegressionL1(C=1.0)\nmodel.fit(X_train, y_train)",
            "from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)\nmodel.fit(X_train, y_train)",
            "from sklearn.linear_model import Lasso\n\nmodel = Lasso(alpha=1.0)\nmodel.fit(X_train, y_train)",
            "from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(regularization='l1', C=1.0)\nmodel.fit(X_train, y_train)"
        ],
        correctIndex: 1,
        explanation: "La deuxième option est correcte pour implémenter une régression logistique avec régularisation L1 (Lasso) en scikit-learn. La classe LogisticRegression accepte le paramètre penalty pour spécifier le type de régularisation ('l1' pour Lasso, 'l2' pour Ridge). Avec penalty='l1', il est nécessaire de spécifier un solver compatible avec la régularisation L1, comme 'liblinear' ou 'saga'. Le paramètre C contrôle la force de la régularisation - c'est l'inverse de la régularisation, donc une valeur plus petite de C signifie une régularisation plus forte. Les autres options sont incorrectes : 'LogisticRegressionL1' n'existe pas dans scikit-learn, la classe Lasso est pour la régression linéaire (et non logistique) avec pénalité L1, et le paramètre 'regularization' n'existe pas (c'est 'penalty' qui est utilisé).",
        difficulty: "medium"
    },

    // Question 7
    {
        question: "Comment imputer correctement les valeurs manquantes numériques et catégorielles dans un DataFrame pandas avec scikit-learn ?",
        options: [
            "# Imputation simple\ndf.fillna(df.mean(), inplace=True)  # Pour les valeurs numériques\ndf.fillna(df.mode().iloc[0], inplace=True)  # Pour les valeurs catégorielles",
            "from sklearn.impute import SimpleImputer\nimport numpy as np\n\n# Pour les colonnes numériques\nnum_imputer = SimpleImputer(strategy='mean')\ndf[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])\n\n# Pour les colonnes catégorielles\ncat_imputer = SimpleImputer(strategy='most_frequent')\ndf[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])",
            "from sklearn.impute import SimpleImputer, KNNImputer\n\n# Pour les colonnes numériques\nnum_imputer = KNNImputer(n_neighbors=5)\ndf[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])\n\n# Pour les colonnes catégorielles\ncat_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')\ndf[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])",
            "import pandas as pd\nfrom sklearn.preprocessing import Imputer\n\n# Créer les imputers\nnum_imputer = Imputer(strategy='mean')\ncat_imputer = Imputer(strategy='most_frequent')\n\n# Appliquer les imputers\ndf[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(df[numerical_cols]), columns=numerical_cols)\ndf[categorical_cols] = pd.DataFrame(cat_imputer.fit_transform(df[categorical_cols]), columns=categorical_cols)"
        ],
        correctIndex: 1,
        explanation: "La deuxième option est correcte pour imputer les valeurs manquantes numériques et catégorielles avec scikit-learn. Elle utilise la classe SimpleImputer, qui remplace les valeurs manquantes par une stratégie spécifiée. Pour les variables numériques, 'mean' remplace par la moyenne de la colonne, tandis que pour les variables catégorielles, 'most_frequent' remplace par la valeur la plus fréquente (mode). Les autres options sont problématiques : la première utilise directement pandas et non scikit-learn comme demandé, la troisième utilise KNNImputer pour les numériques (ce qui peut être valide mais est plus complexe que nécessaire) et remplace les catégorielles par une constante 'MISSING' (ce qui n'est généralement pas optimal), et la quatrième utilise la classe Imputer qui est obsolète dans les versions récentes de scikit-learn (remplacée par SimpleImputer).",
        difficulty: "medium"
    },
    // Question 8 (suite)
    {
        question: "Comment gère-t-on le problème de l'asymétrie des classes (class imbalance) en classification avec scikit-learn ?",
        options: [
            "from sklearn.utils import class_weight\n\n# Calculer les poids des classes\nclass_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)\nclass_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}\n\n# Utiliser les poids dans le modèle\nmodel = RandomForestClassifier(class_weight=class_weight_dict)\nmodel.fit(X_train, y_train)",
            "from sklearn.preprocessing import balance_classes\n\n# Équilibrer les classes\nX_balanced, y_balanced = balance_classes(X_train, y_train, method='smote')\n\n# Entraîner sur les données équilibrées\nmodel = RandomForestClassifier()\nmodel.fit(X_balanced, y_balanced)",
            "from imblearn.over_sampling import SMOTE\n\n# Appliquer SMOTE pour suréchantillonner la classe minoritaire\nsmote = SMOTE(random_state=42)\nX_resampled, y_resampled = smote.fit_resample(X_train, y_train)\n\n# Entraîner sur les données rééchantillonnées\nmodel = RandomForestClassifier()\nmodel.fit(X_resampled, y_resampled)",
            "# Définir un seuil de décision personnalisé\nmodel = RandomForestClassifier()\nmodel.fit(X_train, y_train)\n\n# Ajuster le seuil de décision pour favoriser la classe minoritaire\ny_scores = model.predict_proba(X_test)[:, 1]\ny_pred = (y_scores > 0.3)  # Seuil plus bas que 0.5 par défaut"
        ],
        correctIndex: 2,
        explanation: "La troisième option montre correctement comment gérer le déséquilibre des classes en utilisant SMOTE (Synthetic Minority Over-sampling Technique) de la bibliothèque imbalanced-learn. SMOTE crée des exemples synthétiques de la classe minoritaire en interpolant entre des exemples existants, ce qui est généralement plus efficace qu'un simple suréchantillonnage avec remplacement. La première option utilise une approche différente mais également valide : ajuster les poids des classes inversement proportionnellement à leur fréquence, ce qui fait que le modèle pénalise davantage les erreurs sur la classe minoritaire. La quatrième option montre une troisième approche : ajuster le seuil de décision pour favoriser la détection de la classe minoritaire. La deuxième option est incorrecte car il n'existe pas de fonction balance_classes dans sklearn.preprocessing.",
        difficulty: "medium"
    },

    // Question 9
    {
        question: "Comment gérer les variables catégorielles dans scikit-learn pour les algorithmes qui nécessitent des entrées numériques ?",
        options: [
            "# Utiliser pandas pour l'encodage one-hot\nX_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)",
            "from sklearn.preprocessing import LabelEncoder\n\n# Encoder chaque colonne catégorielle\nle = LabelEncoder()\nfor col in categorical_cols:\n    X[col] = le.fit_transform(X[col])",
            "from sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import OneHotEncoder\n\n# Créer un transformateur pour les colonnes catégorielles\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('cat', OneHotEncoder(drop='first'), categorical_cols)\n    ],\n    remainder='passthrough'  # Laisser les autres colonnes inchangées\n)\n\n# Appliquer la transformation\nX_processed = preprocessor.fit_transform(X)",
            "# Convertir manuellement en variables numériques\nfor col in categorical_cols:\n    unique_values = X[col].unique()\n    for i, value in enumerate(unique_values):\n        X[col] = X[col].replace(value, i)"
        ],
        correctIndex: 2,
        explanation: "La troisième option illustre la méthode recommandée pour gérer les variables catégorielles dans scikit-learn en utilisant ColumnTransformer avec OneHotEncoder. Cette approche permet d'intégrer proprement le prétraitement dans un pipeline scikit-learn et de traiter différemment les colonnes catégorielles et numériques. OneHotEncoder transforme chaque valeur catégorielle en un vecteur binaire où une seule composante est active (1) et les autres sont inactives (0). L'option drop='first' élimine la première catégorie pour éviter la multicolinéarité. Les autres approches sont moins idéales : la première utilise pandas et non le pipeline scikit-learn, rendant plus difficile l'application cohérente de la transformation aux données de test ; la deuxième utilise LabelEncoder qui crée une représentation ordinale inadaptée à la plupart des algorithmes (sauf les arbres) car elle introduit une fausse relation d'ordre ; la quatrième est similaire à LabelEncoder mais implémentée manuellement et sans intégration avec scikit-learn.",
        difficulty: "medium"
    },

    // Question 10
    {
        question: "Comment créer un pipeline scikit-learn avec preprocessing et modélisation ?",
        options: [
            "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n\npipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('classifier', RandomForestClassifier())\n])\n\npipeline.fit(X_train, y_train)\ny_pred = pipeline.predict(X_test)",
            "import sklearn\n\n# Créer un pipeline séquentiel\npipeline = sklearn.pipeline([\n    sklearn.preprocessing.StandardScaler(),\n    sklearn.ensemble.RandomForestClassifier()\n])\n\npipeline.train(X_train, y_train)\ny_pred = pipeline.predict(X_test)",
            "from sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Prétraitement puis modélisation\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)\n\nclf = RandomForestClassifier()\nclf.fit(X_train_scaled, y_train)\ny_pred = clf.predict(X_test_scaled)",
            "from sklearn.Pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Créer un pipeline avec make_pipeline\npipeline = make_pipeline(\n    StandardScaler,\n    RandomForestClassifier\n)\n\npipeline.fit(X_train, y_train)\ny_pred = pipeline.predict(X_test)"
        ],
        correctIndex: 0,
        explanation: "La première option montre correctement comment créer un pipeline scikit-learn, qui enchaîne séquentiellement plusieurs étapes de traitement. La classe Pipeline du module sklearn.pipeline permet de définir ces étapes comme une liste de tuples (nom, transformateur/estimateur). L'avantage du pipeline est qu'il garantit que les mêmes transformations sont appliquées de manière cohérente aux données d'entraînement et de test, et simplifie le flux de travail en permettant d'appeler fit() et predict() sur l'ensemble du pipeline. La troisième option est techniquement correcte mais n'utilise pas de pipeline, ce qui peut entraîner des erreurs lors de l'application des transformations. Les options deux et quatre contiennent des erreurs : il n'existe pas de module sklearn.pipeline (mais sklearn.pipeline), la méthode s'appelle fit() et non train(), et make_pipeline prend des instances d'objets et non des classes.",
        difficulty: "easy"
    },

    // Question 11
    {
        question: "Comment gérer le problème de vanishing gradient dans un réseau de neurones avec TensorFlow/Keras ?",
        options: [
            "# Utiliser des fonctions d'activation adaptées\nmodel = keras.Sequential([\n    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),\n    keras.layers.BatchNormalization(),\n    keras.layers.Dense(64, activation='relu'),\n    keras.layers.BatchNormalization(),\n    keras.layers.Dense(num_classes, activation='softmax')\n])",
            "# Augmenter le taux d'apprentissage\noptimizer = keras.optimizers.Adam(learning_rate=0.1)\nmodel.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])",
            "# Réduire la profondeur du réseau\nmodel = keras.Sequential([\n    keras.layers.Dense(32, activation='sigmoid', input_shape=(input_dim,)),\n    keras.layers.Dense(num_classes, activation='softmax')\n])",
            "# Utiliser un optimizer avec momentum\noptimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)\nmodel.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])"
        ],
        correctIndex: 0,
        explanation: "La première option illustre correctement plusieurs techniques pour gérer le problème de vanishing gradient dans un réseau de neurones. Elle utilise : (1) la fonction d'activation ReLU, qui a une dérivée de 1 pour toutes les entrées positives, évitant ainsi l'atténuation du gradient contrairement aux fonctions sigmoid ou tanh ; (2) la normalisation par lots (BatchNormalization), qui aide à stabiliser la distribution des activations à travers les couches, facilitant la propagation du gradient ; (3) une architecture avec des connexions directes entre couches non adjacentes (skip connections) qui permettent au gradient de contourner certaines couches. Les autres options sont problématiques : augmenter le taux d'apprentissage peut aggraver le problème ou conduire à des instabilités ; réduire la profondeur du réseau et utiliser sigmoid évite le problème mais limite la capacité d'apprentissage ; et bien que l'utilisation du momentum puisse aider, cela ne résout pas directement le problème fondamental.",
        difficulty: "hard"
    },

    // Question 12
    {
        question: "Comment implémenter la validation croisée stratifiée avec recherche d'hyperparamètres pour un classifieur de forêt aléatoire ?",
        options: [
            "from sklearn.model_selection import cross_val_score\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Définir différentes combinaisons d'hyperparamètres\nparams = [\n    {'n_estimators': 100, 'max_depth': 10},\n    {'n_estimators': 200, 'max_depth': 15},\n    {'n_estimators': 300, 'max_depth': 20}\n]\n\n# Tester chaque combinaison\nbest_score = 0\nbest_params = None\nfor p in params:\n    model = RandomForestClassifier(**p)\n    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()\n    if score > best_score:\n        best_score = score\n        best_params = p",
            "from sklearn.model_selection import GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Définir la grille d'hyperparamètres\nparam_grid = {\n    'n_estimators': [100, 200, 300],\n    'max_depth': [10, 15, 20, None]\n}\n\n# Recherche sur grille avec validation croisée\nmodel = RandomForestClassifier()\ngrid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')\ngrid_search.fit(X, y)\n\n# Obtenir les meilleurs paramètres\nbest_params = grid_search.best_params_\nbest_model = grid_search.best_estimator_",
            "from sklearn.model_selection import StratifiedKFold, GridSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\n\n# Définir la grille d'hyperparamètres\nparam_grid = {\n    'n_estimators': [100, 200, 300],\n    'max_depth': [10, 15, 20, None]\n}\n\n# Validation croisée stratifiée\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\n# Recherche sur grille avec validation croisée stratifiée\nmodel = RandomForestClassifier()\ngrid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')\ngrid_search.fit(X, y)\n\n# Obtenir les meilleurs paramètres\nbest_params = grid_search.best_params_\nbest_model = grid_search.best_estimator_",
            "from sklearn.model_selection import RandomizedSearchCV\nfrom sklearn.ensemble import RandomForestClassifier\nfrom scipy.stats import randint\n\n# Définir les distributions d'hyperparamètres\nparam_dist = {\n    'n_estimators': randint(50, 500),\n    'max_depth': randint(5, 30)\n}\n\n# Recherche aléatoire avec validation croisée\nmodel = RandomForestClassifier()\nrandom_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, scoring='accuracy')\nrandom_search.fit(X, y)\n\n# Obtenir les meilleurs paramètres\nbest_params = random_search.best_params_\nbest_model = random_search.best_estimator_"
        ],
        correctIndex: 2,
        explanation: "La troisième option implémente correctement la validation croisée stratifiée avec recherche d'hyperparamètres. Elle utilise StratifiedKFold pour assurer que chaque fold contient approximativement la même proportion de classes que l'ensemble complet, ce qui est crucial pour les données déséquilibrées. Cette stratification est combinée avec GridSearchCV pour explorer systématiquement toutes les combinaisons d'hyperparamètres spécifiées. La première option implémente manuellement la recherche d'hyperparamètres, ce qui est moins efficace et n'utilise pas la validation stratifiée. La deuxième option utilise GridSearchCV mais sans spécifier explicitement la validation stratifiée (bien que GridSearchCV utilise StratifiedKFold par défaut pour les problèmes de classification, il est préférable de le spécifier pour plus de clarté). La quatrième option utilise RandomizedSearchCV qui est une alternative valide à GridSearchCV (explorant aléatoirement l'espace des hyperparamètres plutôt que systématiquement), mais sans validation stratifiée explicite.",
        difficulty: "hard"
    },

    // Question 13
    {
        question: "Comment implémenter une régression logistique multinomiale en scikit-learn ?",
        options: [
            "from sklearn.linear_model import MultinomialLogisticRegression\n\nmodel = MultinomialLogisticRegression()\nmodel.fit(X_train, y_train)",
            "from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(multi_class='ovr')\nmodel.fit(X_train, y_train)",
            "from sklearn.multiclass import OneVsRestClassifier\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = OneVsRestClassifier(LogisticRegression())\nmodel.fit(X_train, y_train)",
            "from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(multi_class='multinomial', solver='lbfgs')\nmodel.fit(X_train, y_train)"
        ],
        correctIndex: 3,
        explanation: "La quatrième option implémente correctement une régression logistique multinomiale en scikit-learn. Le paramètre multi_class='multinomial' spécifie que la perte est minimisée pour toutes les classes simultanément (vraie régression logistique multinomiale), par opposition à une approche one-vs-rest. Il est important de noter que la régression logistique multinomiale nécessite un solver qui la prend en charge, comme 'lbfgs', 'sag', 'saga' ou 'newton-cg'. La première option est incorrecte car il n'existe pas de classe MultinomialLogisticRegression dans scikit-learn. Les options deux et trois implémentent une approche one-vs-rest, où un classifieur binaire séparé est entraîné pour chaque classe (contre toutes les autres), ce qui est différent d'une vraie régression logistique multinomiale.",
        difficulty: "medium"
    },

    // Question 14
    {
        question: "Comment traiter des données de séries temporelles avec des caractéristiques saisonnières en scikit-learn ?",
        options: [
            "from sklearn.preprocessing import MinMaxScaler\nfrom sklearn.linear_model import LinearRegression\n\n# Normaliser les données\nscaler = MinMaxScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Entrainer un modèle de régression\nmodel = LinearRegression()\nmodel.fit(X_scaled, y)",
            "from sklearn.decomposition import PCA\nfrom sklearn.ensemble import RandomForestRegressor\n\n# Réduire la dimensionnalité\npca = PCA(n_components=5)\nX_reduced = pca.fit_transform(X)\n\n# Entrainer un modèle d'ensemble\nmodel = RandomForestRegressor()\nmodel.fit(X_reduced, y)",
            "from sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\n\n# Standardiser les données\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\n# Ajouter des caractéristiques saisonnières\ndef add_seasonal_features(X, period=12):\n    n_samples, n_features = X.shape\n    X_seasonal = np.zeros((n_samples, n_features + 2*period))\n    X_seasonal[:, :n_features] = X\n    \n    # Ajouter sin et cos pour capturer la saisonnalité\n    for p in range(period):\n        X_seasonal[:, n_features + p] = np.sin(2 * np.pi * (p+1) * np.arange(n_samples) / period)\n        X_seasonal[:, n_features + period + p] = np.cos(2 * np.pi * (p+1) * np.arange(n_samples) / period)\n    \n    return X_seasonal\n\nX_with_seasonal = add_seasonal_features(X_scaled)\n\n# Entrainer un modèle régularisé\nmodel = Ridge(alpha=1.0)\nmodel.fit(X_with_seasonal, y)",
            "from statsmodels.tsa.seasonal import seasonal_decompose\nfrom sklearn.ensemble import GradientBoostingRegressor\n\n# Décomposer la série temporelle\nresult = seasonal_decompose(pd.Series(y), model='multiplicative')\ntrend = result.trend\nseasonal = result.seasonal\nresidual = result.resid\n\n# Entrainer un modèle sur les résidus\nmodel = GradientBoostingRegressor()\nmodel.fit(X, residual.dropna())"
        ],
        correctIndex: 2,
        explanation: "La troisième option implémente correctement une approche pour traiter des données de séries temporelles avec des caractéristiques saisonnières en scikit-learn. Elle commence par standardiser les données, puis ajoute explicitement des caractéristiques saisonnières en utilisant des transformations sinusoïdales et cosinusoïdales (encodage circulaire) pour différentes périodes. Cette approche permet de capturer les motifs cycliques sans avoir à les apprendre directement à partir des données, ce qui est particulièrement utile pour les modèles linéaires. La fonction Ridge est utilisée pour la régression avec régularisation L2, ce qui aide à gérer la multicolinéarité potentielle introduite par les nouvelles caractéristiques. Les autres options ne traitent pas spécifiquement la saisonnalité : la première normalise simplement les données et utilise une régression linéaire standard ; la deuxième applique PCA pour la réduction de dimensionnalité, ce qui peut perdre les informations saisonnières ; et la quatrième utilise statsmodels pour décomposer la série puis modélise uniquement les résidus, ce qui n'est pas une approche standard avec scikit-learn.",
        difficulty: "hard"
    },

    // Question 15
    {
        question: "Comment implémenter un modèle multi-output avec scikit-learn (prédiction simultanée de plusieurs variables cibles) ?",
        options: [
            "from sklearn.multioutput import MultiOutputRegressor\nfrom sklearn.ensemble import RandomForestRegressor\n\n# Créer un régresseur multi-output\nmulti_output_model = MultiOutputRegressor(RandomForestRegressor())\nmulti_output_model.fit(X_train, y_train_multi)  # y_train_multi a plusieurs colonnes",
            "from sklearn.ensemble import RandomForestRegressor\n\n# Les modèles d'arbre supportent nativement la régression multi-output\nmodel = RandomForestRegressor()\nmodel.fit(X_train, y_train_multi)  # y_train_multi a plusieurs colonnes",
            "# Créer un modèle séparé pour chaque sortie\nmodels = []\nfor i in range(y_train_multi.shape[1]):\n    model = RandomForestRegressor()\n    model.fit(X_train, y_train_multi[:, i])\n    models.append(model)\n\n# Fonction de prédiction combinée\ndef predict_multi(X):\n    predictions = np.column_stack([model.predict(X) for model in models])\n    return predictions",
            "from sklearn.linear_model import Ridge\n\n# Créer un modèle Ridge pour la régression multi-output\nmodel = Ridge(alpha=1.0)\nmodel.fit(X_train, y_train_multi)  # y_train_multi a plusieurs colonnes"
        ],
        correctIndex: 1,
        explanation: "La deuxième option est correcte car les modèles basés sur des arbres dans scikit-learn, comme RandomForestRegressor, supportent nativement la régression multi-output sans nécessiter de wrapper spécial. Cela signifie qu'ils peuvent être entraînés directement sur une matrice cible y où chaque ligne correspond à un échantillon et chaque colonne à une variable cible différente. La première option utilise MultiOutputRegressor, qui est un wrapper pour adapter les estimateurs qui ne supportent pas nativement les cibles multidimensionnelles. Bien que cette approche fonctionnerait, elle est redondante pour RandomForestRegressor qui supporte déjà cette fonctionnalité. La troisième option entraîne des modèles séparés pour chaque sortie, ce qui est une approche valide mais moins efficace et ne capture pas les corrélations entre les sorties. La quatrième option est également correcte car Ridge supporte aussi nativement la régression multi-output, mais le choix de RandomForestRegressor est généralement préférable pour des problèmes plus complexes.",
        difficulty: "medium"
    },

    // Question 16
    {
        question: "Comment implémenter un encodeur-décodeur (autoencoder) simple avec Keras ?",
        options: [
            "import keras\nfrom keras.models import Sequential\nfrom keras.layers import Dense\n\n# Autoencoder séquentiel\nautoencoder = Sequential([\n    # Encodeur\n    Dense(128, activation='relu', input_shape=(input_dim,)),\n    Dense(64, activation='relu'),\n    Dense(32, activation='relu'),\n    # Décodeur\n    Dense(64, activation='relu'),\n    Dense(128, activation='relu'),\n    Dense(input_dim, activation='sigmoid')\n])\n\nautoencoder.compile(optimizer='adam', loss='mse')\nautoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))",
            "import tensorflow as tf\n\n# Créer l'encodeur\nencode = tf.keras.layers.Dense(32, activation='relu')\n\n# Créer le décodeur\ndecode = tf.keras.layers.Dense(input_dim, activation='sigmoid')\n\n# Fonction d'autoencoder\ndef autoencoder(x):\n    encoded = encode(x)\n    decoded = decode(encoded)\n    return decoded\n\n# Compiler et entraîner\noptimizer = tf.keras.optimizers.Adam()\nloss_fn = tf.keras.losses.MeanSquaredError()\n\n# Entraînement\nfor epoch in range(50):\n    for x_batch in tf.data.Dataset.from_tensor_slices(X_train).batch(256):\n        with tf.GradientTape() as tape:\n            reconstructed = autoencoder(x_batch)\n            loss = loss_fn(x_batch, reconstructed)\n        grads = tape.gradient(loss, [encode.variables, decode.variables])\n        optimizer.apply_gradients(zip(grads, [encode.variables, decode.variables]))",
            "from tensorflow import keras\nfrom tensorflow.keras import layers\n\n# Définir les dimensions\nlatent_dim = 32\n\n# Définir l'encodeur\nencoderInput = keras.Input(shape=(input_dim,))\nx = layers.Dense(128, activation='relu')(encoderInput)\nx = layers.Dense(64, activation='relu')(x)\nencoderOutput = layers.Dense(latent_dim, activation='relu')(x)\nencoder = keras.Model(encoderInput, encoderOutput, name='encoder')\n\n# Définir le décodeur\ndecoderInput = keras.Input(shape=(latent_dim,))\nx = layers.Dense(64, activation='relu')(decoderInput)\nx = layers.Dense(128, activation='relu')(x)\ndecoderOutput = layers.Dense(input_dim, activation='sigmoid')(x)\ndecoder = keras.Model(decoderInput, decoderOutput, name='decoder')\n\n# Définir l'autoencoder\nautoencoder_input = keras.Input(shape=(input_dim,))\nencoded = encoder(autoencoder_input)\ndecoded = decoder(encoded)\nautoencoder = keras.Model(autoencoder_input, decoded, name='autoencoder')\n\n# Compiler et entraîner\nautoencoder.compile(optimizer='adam', loss='mse')\nautoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_test, X_test))",
            "import torch\nimport torch.nn as nn\n\n# Définition de l'autoencoder\nclass Autoencoder(nn.Module):\n    def __init__(self, input_dim, latent_dim=32):\n        super(Autoencoder, self).__init__()\n        \n        # Encodeur\n        self.encoder = nn.Sequential(\n            nn.Linear(input_dim, 128),\n            nn.ReLU(),\n            nn.Linear(128, 64),\n            nn.ReLU(),\n            nn.Linear(64, latent_dim),\n            nn.ReLU()\n        )\n        \n        # Décodeur\n        self.decoder = nn.Sequential(\n            nn.Linear(latent_dim, 64),\n            nn.ReLU(),\n            nn.Linear(64, 128),\n            nn.ReLU(),\n            nn.Linear(128, input_dim),\n            nn.Sigmoid()\n        )\n        \n    def forward(self, x):\n        encoded = self.encoder(x)\n        decoded = self.decoder(encoded)\n        return decoded\n\n# Instancier et entraîner le modèle\nmodel = Autoencoder(input_dim)\noptimizer = torch.optim.Adam(model.parameters())\ncriterion = nn.MSELoss()\n\n# Entraînement (à compléter)"
        ],
        correctIndex: 2,
        explanation: "La troisième option implémente correctement un autoencoder avec l'API fonctionnelle de Keras, qui est plus flexible que l'API séquentielle. Elle sépare explicitement l'encodeur et le décodeur en modèles distincts, ce qui est utile si l'on souhaite les utiliser séparément plus tard (par exemple, pour extraire les représentations latentes ou générer de nouvelles données à partir du code latent). L'approche définit d'abord les modèles d'encodeur et de décodeur, puis les combine pour former l'autoencoder complet. La première option utilise un modèle séquentiel qui fonctionne mais ne permet pas d'accéder facilement aux parties intermédiaires. La deuxième option utilise l'API de bas niveau de TensorFlow avec GradientTape, ce qui est inutilement complexe pour un autoencoder simple. La quatrième option utilise PyTorch au lieu de Keras/TensorFlow, ce qui ne répond pas à la question demandée.",
        difficulty: "hard"
    },

    // Question 17
    Question 17: Visualisation de l'importance des caractéristiques
    {
        question: "Comment visualiser l'importance des caractéristiques d'un modèle d'arbre de décision avec matplotlib ?",
        options: [
            "import matplotlib.pyplot as plt\nimport numpy as np\n\ndef plot_feature_importance(model, feature_names):\n    # Récupérer l'importance des caractéristiques\n    importances = model.feature_importances_\n    \n    # Trier les caractéristiques par importance\n    indices = np.argsort(importances)[::-1]\n    \n    # Barplot\n    plt.figure(figsize=(10, 6))\n    plt.bar(range(len(importances)), importances[indices], align='center')\n    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)\n    plt.xlabel('Caractéristiques')\n    plt.ylabel('Importance')\n    plt.title('Importance des caractéristiques')\n    plt.tight_layout()\n    plt.show()",
            "import seaborn as sns\n\ndef plot_feature_importance(model, feature_names):\n    importances = model.feature_importances_\n    indices = np.argsort(importances)[::-1]\n    \n    plt.figure(figsize=(10, 6))\n    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])\n    plt.xlabel('Importance')\n    plt.ylabel('Caractéristiques')\n    plt.title('Importance des caractéristiques')\n    plt.tight_layout()\n    plt.show()",
            "import plotly.express as px\nimport pandas as pd\n\ndef plot_feature_importance(model, feature_names):\n    importances = model.feature_importances_\n    indices = np.argsort(importances)[::-1]\n    \n    df = pd.DataFrame({\n        'feature': [feature_names[i] for i in indices],\n        'importance': importances[indices]\n    })\n    \n    fig = px.bar(df, x='feature', y='importance', \n                 title='Importance des caractéristiques')\n    fig.show()",
            "import matplotlib.pyplot as plt\nimport numpy as np\n\ndef plot_feature_importance(model, feature_names):\n    importances = model.feature_importances_\n    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)\n    indices = np.argsort(importances)[::-1]\n    \n    plt.figure(figsize=(10, 6))\n    plt.title('Importance des caractéristiques avec écart-type')\n    plt.bar(range(len(importances)), importances[indices], \n            yerr=std[indices], align='center')\n    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)\n    plt.xlabel('Caractéristiques')\n    plt.ylabel('Importance')\n    plt.tight_layout()\n    plt.show()"
        ],
        correctIndex: 3,
        explanation: "La dernière option est la plus complète car elle non seulement visualise l'importance des caractéristiques via un bar plot, mais ajoute également les barres d'erreur représentant l'écart-type de l'importance entre les différents arbres du modèle (utile pour les modèles d'ensemble comme Random Forest). Elle récupère l'écart-type en calculant la variance de l'importance des caractéristiques à travers tous les arbres individuels du modèle.",
        difficulty: "medium"
    },
    {
        question: "Quelle est la principale différence entre la descente de gradient standard et la rétropropagation dans les réseaux de neurones ?",
        options: [
            "La descente de gradient calcule les gradients couche par couche, tandis que la rétropropagation calcule les gradients de la dernière couche vers la première.",
            "La descente de gradient calcule les gradients de la dernière couche vers la première, tandis que la rétropropagation calcule les gradients couche par couche.",
            "Il n'y a pas de différence, les deux méthodes sont identiques.",
            "La rétropropagation est utilisée uniquement pour les réseaux de neurones convolutifs (CNN)."
        ],
        correctIndex: 0,
        explanation: "La rétropropagation est un algorithme qui calcule les gradients des paramètres couche par couche, en partant de la dernière couche et en remontant vers la première, en utilisant la règle de la chaîne. Cela permet de propager l'erreur de la couche de sortie vers les couches précédentes et de mettre à jour les poids du réseau de manière efficace.",
        difficulty: "medium"
    },

    // Question 19: Différence entre Ridge et Lasso
    {
        question: "Quelles sont les principales différences entre la régularisation Ridge (L2) et Lasso (L1) ?",
        options: [
            "Ridge ajoute une pénalité proportionnelle au carré des poids, Lasso ajoute une pénalité proportionnelle à la valeur absolue des poids.",
            "Lasso force certains poids à zéro, Ridge réduit tous les poids proportionnellement.",
            "Ridge et Lasso ont exactement le même effet de régularisation.",
            "Ridge est utilisé pour la classification, Lasso pour la régression."
        ],
        correctIndex: 0,
        explanation: "Ridge (L2) ajoute une pénalité proportionnelle au carré des poids (||w||²), ce qui tend à réduire uniformément la magnitude des poids. Lasso (L1) ajoute une pénalité proportionnelle à la valeur absolue des poids (||w||), ce qui peut forcer certains poids à devenir exactement zéro, réalisant ainsi une sélection de caractéristiques.",
        difficulty: "medium"
    },

    // Question 20: XGBoost vs Random Forest
    {
        question: "Quelles sont les principales différences entre XGBoost et Random Forest ?",
        options: [
            "XGBoost est un algorithme de boosting, Random Forest est un algorithme de bagging.",
            "Random Forest construit des arbres indépendants, XGBoost construit des arbres séquentiellement en corrigeant les erreurs des arbres précédents.",
            "XGBoost ne peut pas gérer les données catégorielles, Random Forest le peut.",
            "Random Forest utilise la régularisation L1, XGBoost utilise la régularisation L2."
        ],
        correctIndex: 1,
        explanation: "XGBoost (eXtreme Gradient Boosting) est un algorithme de boosting qui construit des arbres séquentiellement, chaque nouvel arbre essayant de corriger les erreurs des arbres précédents. Random Forest, en revanche, construit des arbres de décision indépendants en parallèle et agrège leurs prédictions.",
        difficulty: "medium"
    },

    // Question 21: MSE et fonction de coût
    {
        question: "Quelle est la signification de l'erreur quadratique moyenne (MSE) comme fonction de coût ?",
        options: [
            "MSE mesure la précision binaire des prédictions.",
            "MSE calcule la moyenne des carrés des différences entre les prédictions et les valeurs réelles.",
            "MSE est utilisée uniquement pour les problèmes de classification.",
            "MSE pénalise plus fortement les erreurs importantes que les erreurs mineures."
        ],
        correctIndex: 1,
        explanation: "La Mean Squared Error (MSE) calcule la moyenne des carrés des différences entre les valeurs prédites et les valeurs réelles. Le carré permet de pénaliser les erreurs importantes plus fortement et d'éliminer les signes négatifs, ce qui la rend utile pour mesurer la précision des modèles de régression.",
        difficulty: "easy"
    },

    // Question 22: Explosion du gradient
    {
        question: "Comment peut-on atténuer le problème de l'explosion du gradient dans les réseaux de neurones profonds ?",
        options: [
            "En augmentant le taux d'apprentissage.",
            "En utilisant des fonctions d'activation comme ReLU, en appliquant le gradient clipping, et en utilisant des architectures comme ResNet.",
            "En réduisant le nombre de couches du réseau.",
            "En utilisant uniquement des fonctions d'activation sigmoid."
        ],
        correctIndex: 1,
        explanation: "L'explosion du gradient peut être atténuée par plusieurs techniques : l'utilisation de fonctions d'activation comme ReLU qui permet une meilleure propagation du gradient, le gradient clipping qui limite la magnitude des gradients, et des architectures comme ResNet qui utilisent des connexions résiduelles pour faciliter la propagation du gradient à travers les couches profondes.",
        difficulty: "hard"
    },

    // Question 23: CNN
    {
        question: "Quelles sont les principales couches d'un réseau de neurones convolutif (CNN) ?",
        options: [
            "Couches fully connected uniquement.",
            "Couches de convolution, couches de pooling, et couches fully connected.",
            "Couches de pooling seulement.",
            "Couches de convolution sans aucune autre couche."
        ],
        correctIndex: 1,
        explanation: "Un CNN typique comprend des couches de convolution qui extraient des caractéristiques locales, des couches de pooling qui réduisent la dimensionnalité spatiale, et des couches fully connected à la fin pour la classification ou la régression.",
        difficulty: "medium"
    },

    // Question 24: SVM
    {
        question: "Quel est le principe fondamental des Machines à Vecteurs de Support (SVM) ?",
        options: [
            "Maximiser la distance entre les classes en trouvant l'hyperplan séparateur optimal.",
            "Minimiser la variance entre les classes.",
            "Maximiser le nombre de points de support.",
            "Réduire la dimensionnalité des données."
        ],
        correctIndex: 0,
        explanation: "Les SVM cherchent à trouver l'hyperplan qui sépare le mieux deux classes en maximisant la marge (distance) entre l'hyperplan et les points les plus proches de chaque classe, appelés vecteurs de support.",
        difficulty: "medium"
    },

    // Question 25: Adaboost
    {
        question: "Comment fonctionne l'algorithme Adaboost (Adaptive Boosting) ?",
        options: [
            "En créant des arbres de décision totalement aléatoires.",
            "En donnant plus de poids aux exemples mal classés lors des itérations suivantes.",
            "En sélectionnant aléatoirement des caractéristiques à chaque itération.",
            "En utilisant uniquement des arbres de décision parfaitement profonds."
        ],
        correctIndex: 1,
        explanation: "Adaboost fonctionne en donnant progressivement plus de poids aux exemples mal classés lors des itérations précédentes. Chaque nouvel classifieur se concentre davantage sur les points qui étaient difficiles à classer par les classifieurs précédents.",
        difficulty: "hard"
    }
];

// exporter les questions   
if (typeof module !== 'undefined') {
    module.exports = { questionsCode };
}
