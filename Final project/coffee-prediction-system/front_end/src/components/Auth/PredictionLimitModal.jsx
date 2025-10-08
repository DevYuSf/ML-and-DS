import React from 'react';

const PredictionLimitModal = ({ 
  predictionsUsed, 
  onSignUp, 
  onContinueLater,
  timeRemaining = "20 hours"
}) => {
  return (
    <div className="fixed inset-0 bg-transparent backdrop-blur-sm bg-opacity-50 flex items-center justify-center p-6 z-50">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full text-center">
        {/* Celebration Emoji */}
        <div className="text-6xl mb-4">🎉</div>
        
        {/* Title */}
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          You're a Coffee Expert!
        </h2>
        
        {/* Message */}
        <p className="text-gray-600 mb-2">
          You've used <span className="font-bold text-blue-600">{predictionsUsed}/3</span> free predictions
        </p>
        
        <p className="text-gray-600 mb-6">
          Sign up for unlimited predictions and advanced features!
        </p>
        
        {/* Feature List */}
        <div className="bg-blue-50 rounded-lg p-4 mb-6 text-left">
          <h3 className="font-semibold text-blue-800 mb-2">Premium Features:</h3>
          <ul className="text-blue-700 space-y-1 text-sm">
            <li>✅ Unlimited predictions</li>
            <li>✅ Advanced analytics</li>
            <li>✅ Export predictions</li>
            <li>✅ Priority support</li>
          </ul>
        </div>
        
        {/* Action Buttons */}
        <div className="space-y-3">
          <button
            onClick={onSignUp}
            className="w-full bg-gradient-to-r from-green-500 to-blue-600 text-white py-3 rounded-lg font-semibold hover:shadow-lg transition-all"
          >
            Sign Up for Unlimited Access
          </button>
          
          <button
            onClick={onContinueLater}
            className="w-full border-2 border-gray-300 text-gray-600 py-3 rounded-lg font-semibold hover:bg-gray-50 transition-all"
          >
            Come back in {timeRemaining}
          </button>
        </div>
        
        {/* Note */}
        <p className="text-xs text-gray-500 mt-4">
          Free predictions reset every 20 hours
        </p>
      </div>
    </div>
  );
};

export default PredictionLimitModal;