import { useState, useEffect } from 'react';

const usePredictionCounter = (isUserAuthenticated = false) => {
  // Get prediction count from localStorage
  const getStoredPredictions = () => {
    const stored = localStorage.getItem('brewai_predictions');
    return stored ? JSON.parse(stored) : { count: 0, timestamp: null };
  };

  const [predictionData, setPredictionData] = useState(getStoredPredictions);

  // Check if 20 hours have passed since last prediction
  const shouldResetCount = () => {
    if (!predictionData.timestamp) return false;
    
    const twentyHours = 20 * 60 * 60 * 1000;
    const timeSinceLastPrediction = Date.now() - predictionData.timestamp;
    
    return timeSinceLastPrediction >= twentyHours;
  };

  // Reset count if 20 hours have passed
  useEffect(() => {
    if (shouldResetCount()) {
      setPredictionData({ count: 0, timestamp: null });
    }
  }, []);

  // Save to localStorage whenever predictionData changes
  useEffect(() => {
    localStorage.setItem('brewai_predictions', JSON.stringify(predictionData));
  }, [predictionData]);

  // ✅ FIX: Check if user can make predictions
  const canPredict = () => {
    // If user is authenticated, unlimited predictions!
    if (isUserAuthenticated) {
      return true;
    }
    
    // For non-authenticated users, check limits
    if (shouldResetCount()) {
      setPredictionData({ count: 0, timestamp: null });
      return true;
    }
    
    return predictionData.count < 3;
  };

  // Record a new prediction
  const recordPrediction = () => {
    // ✅ FIX: Don't record if user is authenticated (unlimited)
    if (isUserAuthenticated) {
      return true;
    }
    
    if (canPredict()) {
      setPredictionData({
        count: predictionData.count + 1,
        timestamp: Date.now()
      });
      return true;
    }
    return false;
  };

  // Get remaining predictions
  const getRemainingPredictions = () => {
    // ✅ FIX: Unlimited for authenticated users
    if (isUserAuthenticated) {
      return 'Unlimited';
    }
    
    if (shouldResetCount()) {
      return 3;
    }
    return 3 - predictionData.count;
  };

  // Get time until reset
  const getTimeUntilReset = () => {
    if (!predictionData.timestamp || isUserAuthenticated) return "now";
    
    const twentyHours = 20 * 60 * 60 * 1000;
    const timeSinceLastPrediction = Date.now() - predictionData.timestamp;
    const timeRemaining = twentyHours - timeSinceLastPrediction;
    
    if (timeRemaining <= 0) return "now";
    
    const hours = Math.floor(timeRemaining / (60 * 60 * 1000));
    const minutes = Math.floor((timeRemaining % (60 * 60 * 1000)) / (60 * 1000));
    
    return `${hours}h ${minutes}m`;
  };

  return {
    canPredict,
    recordPrediction,
    predictionsUsed: isUserAuthenticated ? 'Unlimited' : predictionData.count,
    remainingPredictions: getRemainingPredictions(),
    timeUntilReset: getTimeUntilReset(),
    isUnlimited: isUserAuthenticated // ✅ NEW: Track unlimited status
  };
};

export default usePredictionCounter;