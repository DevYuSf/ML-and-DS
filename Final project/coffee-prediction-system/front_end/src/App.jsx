import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import OnboardingFlow from './components/Onboarding/OnboardingFlow';
import LoginForm from './components/Auth/LoginForm';
import RegisterForm from './components/Auth/RegisterForm';
import PredictionLimitModal from './components/Auth/PredictionLimitModal';
import { coffeeAPI } from './services/api';
import usePredictionCounter from './hooks/usePredictionCounter';

// Main App Component (now uses Auth Context)
function AppContent() {
  const { user, isAuthenticated, login, register, logout } = useAuth();
  
  // App state
  const [currentView, setCurrentView] = useState('onboarding');
  const [authMode, setAuthMode] = useState('login');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [selectedModel, setSelectedModel] = useState(null);
  const [showLimitModal, setShowLimitModal] = useState(false);

  // ✅ FIX: Pass authentication status to prediction counter
  const {
    canPredict,
    recordPrediction,
    predictionsUsed,
    remainingPredictions,
    timeUntilReset,
    isUnlimited
  } = usePredictionCounter(isAuthenticated);

  // Check if user has completed onboarding
  useEffect(() => {
    const hasCompletedOnboarding = localStorage.getItem('brewai_onboarding_complete');
    
    // ✅ FIX: If user is authenticated, go directly to dashboard
    if (isAuthenticated) {
      setCurrentView('dashboard');
    } else if (hasCompletedOnboarding) {
      setCurrentView('dashboard');
    } else {
      setCurrentView('onboarding');
    }
  }, [isAuthenticated]);

  // Check API health
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await coffeeAPI.healthCheck();
      setApiStatus('connected');
    } catch (error) {
      setApiStatus('disconnected');
    }
  };

  // Onboarding complete handler
  const handleOnboardingComplete = () => {
    localStorage.setItem('brewai_onboarding_complete', 'true');
    setCurrentView('dashboard');
  };

  // ✅ FIX: Proper authentication handlers
  const handleLogin = (userData) => {
    console.log('Logging in:', userData);
    login(userData);
    setCurrentView('dashboard');
  };

  const handleRegister = (userData) => {
    console.log('Registering:', userData);
    register(userData);
    setCurrentView('dashboard');
  };

  const handleShowAuth = (mode = 'login') => {
    setAuthMode(mode);
    setCurrentView('auth');
  };

  const handleLogout = () => {
    logout();
    setCurrentView('dashboard');
  };

  // Prediction handler
  const handlePredict = async (inputs) => {
    if (!canPredict()) {
      setShowLimitModal(true);
      return;
    }

    setIsLoading(true);
    try {
      const result = await coffeeAPI.getPrediction(inputs, selectedModel);
      setPrediction(result);
      recordPrediction();
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Please check your Flask server.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelSelect = async (modelName) => {
    try {
      await coffeeAPI.selectModel(modelName);
      setSelectedModel(modelName);
    } catch (error) {
      console.error('Model selection failed:', error);
    }
  };

  // Render different views based on currentView state
  const renderCurrentView = () => {
    switch (currentView) {
      case 'onboarding':
        return <OnboardingFlow onComplete={handleOnboardingComplete} />;
      
      case 'auth':
        return authMode === 'login' ? (
          <LoginForm 
            onLogin={handleLogin} 
            onSwitchToRegister={() => setAuthMode('register')} 
          />
        ) : (
          <RegisterForm 
            onRegister={handleRegister} 
            onSwitchToLogin={() => setAuthMode('login')} 
          />
        );
      
      case 'dashboard':
        return (
          <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100">
            {/* ✅ RESPONSIVE HEADER */}
            <div className="bg-white shadow-sm border-b">
              <div className="container mx-auto px-3 sm:px-4 py-3 sm:py-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-4">
                  {/* Left Section - Title & Status */}
                  <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
                    <h1 className="text-xl sm:text-2xl font-bold text-gray-800 flex items-center gap-2">
                      <span className="text-amber-600">☕</span>
                      BrewAI Predictor
                      {isAuthenticated && (
                        <span className="text-xs bg-green-500 text-white px-2 py-1 rounded-full ml-2">
                          Premium
                        </span>
                      )}
                    </h1>
                    
                    {/* Status Badges - Stack on mobile */}
                    <div className="flex flex-wrap gap-2">
                      {/* API Status */}
                      <div className={`px-2 py-1 rounded-full text-xs sm:text-sm ${
                        apiStatus === 'connected' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {apiStatus === 'connected' ? '✅ Connected' : '❌ Disconnected'}
                      </div>

                      {/* Prediction Counter */}
                      <div className={`px-2 py-1 rounded-full text-xs sm:text-sm ${
                        isAuthenticated 
                          ? 'bg-purple-100 text-purple-800' 
                          : 'bg-blue-100 text-blue-800'
                      }`}>
                        {isAuthenticated ? '⭐ Unlimited' : `${remainingPredictions}/3`}
                      </div>
                    </div>
                  </div>

                  {/* Right Section - Controls & Auth */}
                  <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-4">
                    {/* Model Selector - Full width on mobile */}
                    <select 
                      value={selectedModel || ''}
                      onChange={(e) => handleModelSelect(e.target.value)}
                      className="w-full sm:w-auto bg-white border border-gray-300 rounded-lg px-3 py-2 text-sm"
                    >
                      <option value="">Auto (Best)</option>
                      <option value="LogisticRegression">Logistic Regression</option>
                      <option value="RandomForest">Random Forest</option>
                      <option value="XGBoost">XGBoost</option>
                      <option value="NaiveBayes">Naive Bayes</option>
                    </select>

                    {/* Auth Buttons */}
                    {isAuthenticated ? (
                      <div className="flex items-center gap-2 sm:gap-3">
                        <span className="text-xs sm:text-sm text-gray-600 hidden sm:block">
                          Hi, {user?.name || 'User'}!
                        </span>
                        <button
                          onClick={handleLogout}
                          className="flex-1 sm:flex-none bg-gray-500 text-white px-3 sm:px-4 py-2 rounded-lg text-sm font-semibold hover:bg-gray-600"
                        >
                          Logout
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => handleShowAuth('login')}
                        className="flex-1 sm:flex-none bg-amber-500 text-white px-3 sm:px-4 py-2 rounded-lg text-sm font-semibold hover:bg-amber-600"
                      >
                        Sign In
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ✅ RESPONSIVE MAIN CONTENT */}
            <div className="container mx-auto px-3 sm:px-4 py-6 sm:py-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8 max-w-7xl mx-auto">
                <PredictionForm 
                  onPredict={handlePredict}
                  isLoading={isLoading}
                />
                <ResultsDisplay 
                  prediction={prediction}
                  isLoading={isLoading}
                />
              </div>
            </div>

            {/* Prediction Limit Modal */}
            {showLimitModal && !isAuthenticated && (
              <PredictionLimitModal
                predictionsUsed={predictionsUsed}
                onSignUp={() => {
                  setShowLimitModal(false);
                  handleShowAuth('register');
                }}
                onContinueLater={() => setShowLimitModal(false)}
                timeRemaining={timeUntilReset}
              />
            )}
          </div>
        );
      
      default:
        return <OnboardingFlow onComplete={handleOnboardingComplete} />;
    }
  };

  return renderCurrentView();
}

// Main App Wrapper with Auth Provider
export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}