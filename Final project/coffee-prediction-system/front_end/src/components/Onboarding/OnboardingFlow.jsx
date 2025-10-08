import React, { useState } from 'react';
import SplashScreen from './SplashScreen';

const OnboardingFlow = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);

  // Onboarding steps data
  const steps = [
    {
      emoji: "☕",
      title: "Welcome to BrewAI!",
      description: "Predict coffee cravings with AI magic. Know what customers want before they order!",
      showSkip: true
    },
    {
      emoji: "🔮",
      title: "How It Works",
      description: "Select time, day, and month. Our AI predicts popular coffee types and optimal pricing!",
      showSkip: true
    },
    {
      emoji: "🎯",
      title: "Ready to Predict!",
      description: "Get 3 free predictions. Sign up for unlimited access and advanced features!",
      showSkip: false
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handleSkip = () => {
    onComplete();
  };

  const currentStepData = steps[currentStep];

  return (
    <SplashScreen
      {...currentStepData}
      onNext={handleNext}
      onSkip={handleSkip}
      autoNavigate={currentStep < steps.length - 1} // No auto on last step
    />
  );
};

export default OnboardingFlow;