import React, { useState } from 'react';

const PredictionForm = ({ onPredict, isLoading }) => {
  const [inputs, setInputs] = useState({
    hour: new Date().getHours(),
    weekday: 'Mon',
    month: 'Jan'
  });

  // Options for dropdowns
  const hours = Array.from({ length: 24 }, (_, i) => i);
  const weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  const handleInputChange = (field, value) => {
    setInputs(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(inputs);
  };

  const getTimeOfDay = (hour) => {
    if (hour >= 5 && hour < 12) return '🌅 Morning';
    if (hour >= 12 && hour < 17) return '☀️ Afternoon';
    return '🌙 Evening';
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center gap-2">
        <span className="text-amber-600">⏰</span>
        When's Coffee Time?
      </h2>
      <p className="text-gray-600 mb-6">Select the time parameters for prediction</p>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Hour Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Hour of Day <span className="text-amber-600">({getTimeOfDay(inputs.hour)})</span>
          </label>
          <div className="grid grid-cols-6 gap-2">
            {hours.map(hour => (
              <button
                key={hour}
                type="button"
                onClick={() => handleInputChange('hour', hour)}
                className={`p-2 rounded-lg text-sm font-medium transition-all ${
                  inputs.hour === hour
                    ? 'bg-amber-500 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {hour}:00
              </button>
            ))}
          </div>
        </div>

        {/* Weekday Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Day of Week
          </label>
          <div className="flex flex-wrap gap-2">
            {weekdays.map(day => (
              <button
                key={day}
                type="button"
                onClick={() => handleInputChange('weekday', day)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  inputs.weekday === day
                    ? 'bg-amber-500 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {day}
              </button>
            ))}
          </div>
        </div>

        {/* Month Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Month
          </label>
          <div className="grid grid-cols-4 gap-2">
            {months.map(month => (
              <button
                key={month}
                type="button"
                onClick={() => handleInputChange('month', month)}
                className={`p-2 rounded-lg text-sm font-medium transition-all ${
                  inputs.month === month
                    ? 'bg-amber-500 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {month}
              </button>
            ))}
          </div>
        </div>

        {/* Current Selection Display */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h3 className="font-semibold text-amber-800 mb-2">Selected Parameters:</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-amber-600 font-medium">Hour:</span>
              <div className="text-gray-700">{inputs.hour}:00</div>
            </div>
            <div>
              <span className="text-amber-600 font-medium">Day:</span>
              <div className="text-gray-700">{inputs.weekday}</div>
            </div>
            <div>
              <span className="text-amber-600 font-medium">Month:</span>
              <div className="text-gray-700">{inputs.month}</div>
            </div>
          </div>
        </div>

        {/* Predict Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-gradient-to-r from-amber-500 to-orange-500 text-white py-3 px-4 rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <div className="flex items-center justify-center gap-2">
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Predicting...
            </div>
          ) : (
            <div className="flex items-center justify-center gap-2">
              <span>🔮</span>
              Predict Coffee Magic!
            </div>
          )}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm;