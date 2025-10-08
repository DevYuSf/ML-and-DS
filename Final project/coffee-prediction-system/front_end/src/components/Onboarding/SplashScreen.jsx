import React, { useEffect } from 'react';

const SplashScreen = ({ 
  title, 
  description, 
  emoji, 
  onNext, 
  onSkip, 
  showSkip = true,
  autoNavigate = true 
}) => {
  // Auto-navigate after 10 seconds
  useEffect(() => {
    if (autoNavigate) {
      const timer = setTimeout(() => {
        onNext();
      }, 10000);
      return () => clearTimeout(timer);
    }
  }, [onNext, autoNavigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center p-6">
      <div className="text-center text-white max-w-md">
        {/* Emoji/Icon */}
        <div className="text-6xl mb-6 animate-bounce">
          {emoji}
        </div>
        
        {/* Title */}
        <h1 className="text-4xl font-bold mb-4">
          {title}
        </h1>
        
        {/* Description */}
        <p className="text-xl mb-8 opacity-90">
          {description}
        </p>
        
        {/* Progress Dots */}
        <div className="flex justify-center gap-2 mb-8">
          <div className="w-3 h-3 bg-white rounded-full"></div>
          <div className="w-3 h-3 bg-white opacity-50 rounded-full"></div>
          <div className="w-3 h-3 bg-white opacity-50 rounded-full"></div>
        </div>
        
        {/* Action Buttons */}
        <div className="flex gap-4 justify-center">
          {showSkip && (
            <button
              onClick={onSkip}
              className="px-6 py-3 border-2 border-white rounded-full text-white hover:bg-white hover:bg-opacity-20 transition-all"
            >
              Skip
            </button>
          )}
          <button
            onClick={onNext}
            className="px-6 py-3 bg-white text-purple-600 rounded-full font-semibold hover:scale-105 transition-transform"
          >
            Next
          </button>
        </div>
        
        {/* Auto-navigate Timer */}
        {autoNavigate && (
          <div className="mt-6 text-white opacity-70">
            Auto-continue in <span className="font-mono">10</span> seconds...
          </div>
        )}
      </div>
    </div>
  );
};

export default SplashScreen;