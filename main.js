// quiz-main.js
// Script principal du QCM sur l'IA

// Variables globales
let currentQuestions = [];
let currentQuestionIndex = 0;
let score = 0;
let selectedOption = null;
let answeredQuestions = [];
let currentCategory = '';

// Éléments DOM
const categorySelection = document.getElementById('category-selection');
const quizSection = document.getElementById('quiz-section');
const questionContainer = document.getElementById('question-container');
const prevBtn = document.getElementById('prev-btn');
const checkBtn = document.getElementById('check-btn');
const nextBtn = document.getElementById('next-btn');
const feedback = document.getElementById('feedback');
const explanation = document.getElementById('explanation');
const resultsDiv = document.getElementById('results');
const scoreDisplay = document.getElementById('score-display');
const scoreMessage = document.getElementById('score-message');
const restartBtn = document.getElementById('restart-btn');
const categorySelectBtn = document.getElementById('category-select-btn');
const progress = document.getElementById('progress');
const quizContainer = document.getElementById('quiz');
const categoryTitle = document.getElementById('category-title');

// Gérer la sélection de catégorie
document.querySelectorAll('.category-btn').forEach(button => {
    button.addEventListener('click', () => {
        const category = button.getAttribute('data-category');
        currentCategory = category;
        
        switch(category) {
            case 'theoriques':
                currentQuestions = questionsTheoriques;
                categoryTitle.textContent = "Questions Théoriques";
                break;
            case 'code':
                currentQuestions = questionsCode;
                categoryTitle.textContent = "Questions Code Python";
                break;
            case 'complexes':
                currentQuestions = questionsComplexes;
                categoryTitle.textContent = "Questions Complexes";
                break;
            case 'all':
                // Combiner toutes les questions
                currentQuestions = [...questionsTheoriques, ...questionsCode, ...questionsComplexes];
                categoryTitle.textContent = "Toutes les Questions";
                // Mélanger les questions
                shuffleArray(currentQuestions);
                break;
        }
        
        // Masquer la sélection de catégorie et afficher le quiz
        categorySelection.style.display = 'none';
        quizSection.style.display = 'block';
        
        // Initialiser le quiz avec les questions sélectionnées
        initQuiz();
    });
});

// Fonction pour mélanger un tableau (algorithme de Fisher-Yates)
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Retourner à la sélection de catégorie
categorySelectBtn.addEventListener('click', () => {
    quizSection.style.display = 'none';
    categorySelection.style.display = 'block';
    resultsDiv.style.display = 'none';
});

// Initialisation du quiz
function initQuiz() {
    currentQuestionIndex = 0;
    score = 0;
    selectedOption = null;
    answeredQuestions = new Array(currentQuestions.length).fill(false);
    
    showQuestion(currentQuestionIndex);
    updateProgressBar();
    
    quizContainer.style.display = 'block';
    resultsDiv.style.display = 'none';
    prevBtn.disabled = true;
    checkBtn.disabled = true;
    nextBtn.disabled = true;
}

// Afficher une question
function showQuestion(index) {
    const question = currentQuestions[index];
    questionContainer.innerHTML = '';

    const questionDiv = document.createElement('div');
    questionDiv.className = 'question active';
    
    const difficultySpan = document.createElement('span');
    difficultySpan.className = `difficulty ${question.difficulty}`;
    difficultySpan.textContent = question.difficulty.charAt(0).toUpperCase() + question.difficulty.slice(1);
    
    questionDiv.innerHTML = `<h3>Question ${index + 1}/${currentQuestions.length} <span class="difficulty ${question.difficulty}">${difficultySpan.textContent}</span></h3><p>${question.question}</p>`;

    const optionsDiv = document.createElement('div');
    optionsDiv.className = 'options';

    question.options.forEach((option, optionIndex) => {
        const optionDiv = document.createElement('div');
        optionDiv.className = 'option';
        optionDiv.innerHTML = option;

        // Si cette question a déjà été répondue, afficher le résultat
        if (answeredQuestions[index]) {
            if (optionIndex === question.correctIndex) {
                optionDiv.classList.add('correct');
            } else if (optionIndex === selectedOption && optionIndex !== question.correctIndex) {
                optionDiv.classList.add('incorrect');
            }
        } else {
            // Sinon, ajouter les écouteurs d'événements pour la sélection
            optionDiv.addEventListener('click', () => {
                // Retirer la classe 'selected' de toutes les options
                document.querySelectorAll('.option').forEach(opt => {
                    opt.classList.remove('selected');
                });
                
                // Ajouter la classe 'selected' à l'option cliquée
                optionDiv.classList.add('selected');
                selectedOption = optionIndex;
                checkBtn.disabled = false;
            });
        }

        optionsDiv.appendChild(optionDiv);
    });

    questionDiv.appendChild(optionsDiv);
    questionContainer.appendChild(questionDiv);

    // Mettre à jour les boutons
    prevBtn.disabled = index === 0;
    
    if (answeredQuestions[index]) {
        checkBtn.disabled = true;
        nextBtn.disabled = index === currentQuestions.length - 1;
        explanation.innerHTML = question.explanation;
        explanation.style.display = 'block';
    } else {
        checkBtn.disabled = selectedOption === null;
        nextBtn.disabled = true;
        explanation.style.display = 'none';
    }
    
    // Masquer le feedback
    feedback.style.display = 'none';
}

// Vérifier la réponse
function checkAnswer() {
    if (selectedOption === null) return;

    const question = currentQuestions[currentQuestionIndex];
    const options = document.querySelectorAll('.option');

    // Marquer cette question comme répondue
    answeredQuestions[currentQuestionIndex] = true;

    // Afficher les options correctes et incorrectes
    options.forEach((option, index) => {
        if (index === question.correctIndex) {
            option.classList.add('correct');
        } else if (index === selectedOption && index !== question.correctIndex) {
            option.classList.add('incorrect');
        }
    });

    // Mettre à jour le score
    if (selectedOption === question.correctIndex) {
        score++;
        feedback.innerHTML = '<p>Correct! 👍</p>';
        feedback.style.backgroundColor = '#d4edda';
    } else {
        feedback.innerHTML = '<p>Incorrect! La bonne réponse est la suivante.</p>';
        feedback.style.backgroundColor = '#f8d7da';
    }

    // Afficher les explications
    explanation.innerHTML = question.explanation;
    explanation.style.display = 'block';
    feedback.style.display = 'block';

    // Mettre à jour les boutons
    checkBtn.disabled = true;
    nextBtn.disabled = currentQuestionIndex === currentQuestions.length - 1;

    // Si c'est la dernière question et qu'elle est répondue, afficher les résultats
    if (currentQuestionIndex === currentQuestions.length - 1 && answeredQuestions[currentQuestionIndex]) {
        // Vérifier si toutes les questions ont été répondues
        nextBtn.disabled = true;
        
        // Ajouter un bouton pour voir les résultats
        const showResultsBtn = document.createElement('button');
        showResultsBtn.textContent = "Voir les résultats";
        showResultsBtn.className = "show-results-btn";
        showResultsBtn.addEventListener('click', showResults);
        
        const controlsDiv = document.querySelector('.controls');
        if (!document.querySelector('.show-results-btn')) {
            controlsDiv.appendChild(showResultsBtn);
        }
    }
}

// Passer à la question suivante
function nextQuestion() {
    if (currentQuestionIndex < currentQuestions.length - 1) {
        currentQuestionIndex++;
        selectedOption = null;
        showQuestion(currentQuestionIndex);
        updateProgressBar();
    }
}

// Revenir à la question précédente
function prevQuestion() {
    if (currentQuestionIndex > 0) {
        currentQuestionIndex--;
        selectedOption = null;
        showQuestion(currentQuestionIndex);
        updateProgressBar();
    }
}

// Afficher les résultats

function showResults() {
    // Calculer le nombre de questions répondues
    const answeredCount = answeredQuestions.filter(a => a).length;
    
    // Si toutes les questions n'ont pas été répondues, demander confirmation
    if (answeredCount < currentQuestions.length) {
        const confirmFinish = confirm(`Vous n'avez répondu qu'à ${answeredCount} questions sur ${currentQuestions.length}. Voulez-vous vraiment terminer le quiz?`);
        if (!confirmFinish) return;
    }
    
    quizContainer.style.display = 'none';
    resultsDiv.style.display = 'block';
    
    const percentage = Math.round((score / answeredCount) * 100);
    scoreDisplay.innerHTML = `Vous avez obtenu ${score} sur ${answeredCount} (${percentage}%)`;
    
    let recommendationTitle = '';
    let recommendation = '';
    
    if (percentage >= 80) {
        scoreMessage.innerHTML = "Excellent! Vous maîtrisez très bien les concepts avancés d'IA.";
        recommendationTitle = "Niveau Expert 🏆";
        recommendation = `
            <strong>Félicitations!</strong> Votre maîtrise des concepts d'IA est remarquable. 
            Suggestions pour continuer à progresser :
            - Approfondissez vos connaissances en explorant des articles de recherche récents
            - Participez à des compétitions Kaggle ou des projets de recherche
            - Commencez à travailler sur des projets d'IA avancés et complexes
        `;
    } else if (percentage >= 60) {
        scoreMessage.innerHTML = "Bon travail! Vous avez une bonne compréhension des concepts avancés d'IA.";
        recommendationTitle = "Niveau Intermédiaire Avancé 🚀";
        recommendation = `
            <strong>Vous êtes sur la bonne voie!</strong> Quelques recommandations pour progresser :
            - Suivez des cours en ligne avancés sur l'IA (Coursera, edX)
            - Implementez des algorithmes d'IA à partir de zéro
            - Lisez des livres techniques sur l'apprentissage automatique et le deep learning
        `;
    } else if (percentage >= 40) {
        scoreMessage.innerHTML = "Pas mal! Mais il y a encore place à l'amélioration sur les concepts avancés.";
        recommendationTitle = "Niveau Intermédiaire 📚";
        recommendation = `
            <strong>Vous avez des bases solides!</strong> Voici comment approfondir vos connaissances :
            - Revisez les concepts fondamentaux de l'IA
            - Suivez des tutoriels pratiques et des cours en ligne
            - Pratiquez la programmation avec des bibliothèques comme scikit-learn et TensorFlow
            - Travaillez sur des mini-projets d'IA
        `;
    } else {
        scoreMessage.innerHTML = "Continuez à étudier les concepts avancés d'IA pour améliorer votre compréhension.";
        recommendationTitle = "Niveau Débutant 🌱";
        recommendation = `
            <strong>Ne vous découragez pas!</strong> Voici un plan pour progresser :
            - Commencez par des cours d'introduction à l'IA et au machine learning
            - Suivez des tutoriels pas à pas
            - Apprenez les bases des langages Python et R
            - Pratiquez avec des datasets simples et des algorithmes de base
        `;
    }
    
    // Ajouter les recommandations
    const detailsDiv = document.getElementById('results-details');
    detailsDiv.innerHTML = `
        <h3>${recommendationTitle}</h3>
        <p>${recommendation}</p>
        <h4>Détail des réponses :</h4>
    `;
    
    // Afficher le détail des réponses
    for (let i = 0; i < currentQuestions.length; i++) {
        if (answeredQuestions[i]) {
            const questionResult = document.createElement('div');
            questionResult.className = 'question-result';
            
            const isCorrect = currentQuestions[i].correctIndex === selectedOption;
            const resultClass = isCorrect ? 'correct-answer' : 'incorrect-answer';
            
            questionResult.innerHTML = `
                <p class="${resultClass}">Question ${i+1}: ${isCorrect ? 'Correct' : 'Incorrect'}</p>
                <details>
                    <summary>${currentQuestions[i].question.substring(0, 100)}...</summary>
                    <p>${currentQuestions[i].explanation}</p>
                </details>
            `;
            
            detailsDiv.appendChild(questionResult);
        }
    }
}