import React from 'react';

const ResultsDisplay = ({ prediction, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-amber-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Brewing Predictions...</h3>
          <p className="text-gray-600">AI is analyzing coffee patterns</p>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
        <div className="text-center">
          <div className="text-6xl mb-4">☕</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Ready to Predict!</h3>
          <p className="text-gray-600">Select time parameters and click predict to see magical coffee insights</p>
        </div>
      </div>
    );
  }

  const { predictions, recommendations, insights } = prediction;

  // Confidence color based on percentage
  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'text-green-600';
    if (confidence >= 60) return 'text-amber-600';
    return 'text-red-600';
  };

  const getConfidenceBarColor = (confidence) => {
    if (confidence >= 80) return 'bg-green-500';
    if (confidence >= 60) return 'bg-amber-500';
    return 'bg-red-500';
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
          <span className="text-amber-600">✨</span>
          Prediction Results
        </h2>
        <div className="bg-amber-100 text-amber-800 px-3 py-1 rounded-full text-sm font-medium">
          {predictions.model_used}
        </div>
      </div>

      {/* Main Prediction Cards */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Coffee Prediction */}
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-xl p-4 border border-amber-200">
          <h3 className="font-semibold text-amber-800 mb-3 flex items-center gap-2">
            <span>☕</span>
            Coffee Type
          </h3>
          <div className="text-xl font-bold text-gray-800 mb-2">
            {predictions.coffee_group.replace(/_/g, ' ')}
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Confidence:</span>
              <span className={`font-semibold ${getConfidenceColor(predictions.coffee_confidence)}`}>
                {predictions.coffee_confidence}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getConfidenceBarColor(predictions.coffee_confidence)}`}
                style={{ width: `${predictions.coffee_confidence}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Price Prediction */}
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
          <h3 className="font-semibold text-green-800 mb-3 flex items-center gap-2">
            <span>💰</span>
            Price Tier
          </h3>
          <div className="text-xl font-bold text-gray-800 mb-2">
            {predictions.price_tier}
            {predictions.price_tier === 'Luxury' && (
              <span className="text-yellow-500 ml-2">⭐</span>
            )}
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Confidence:</span>
              <span className={`font-semibold ${getConfidenceColor(predictions.price_confidence)}`}>
                {predictions.price_confidence}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${getConfidenceBarColor(predictions.price_confidence)}`}
                style={{ width: `${predictions.price_confidence}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations && (
        <div className="bg-blue-50 rounded-xl p-4 border border-blue-200 mb-6">
          <h3 className="font-semibold text-blue-800 mb-3 flex items-center gap-2">
            <span>🎯</span>
            Recommended Drinks
          </h3>
          <p className="text-blue-700 mb-3">{recommendations.success_message}</p>
          <div className="flex flex-wrap gap-2 mb-3">
            {recommendations.suggested_drinks.map((drink, index) => (
              <span key={index} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                {drink}
              </span>
            ))}
          </div>
          <p className="text-blue-600 text-sm">
            <span className="font-semibold">💡 Tip:</span> {recommendations.preparation_tip}
          </p>
        </div>
      )}

      {/* Business Insights */}
      {insights && insights.length > 0 && (
        <div className="bg-purple-50 rounded-xl p-4 border border-purple-200">
          <h3 className="font-semibold text-purple-800 mb-3 flex items-center gap-2">
            <span>🏪</span>
            Business Insights
          </h3>
          <div className="space-y-2">
            {insights.map((insight, index) => (
              <div key={index} className="flex items-start gap-3 text-purple-700">
                <span className="mt-1">•</span>
                <span>{insight}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;