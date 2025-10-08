import React, { createContext, useState, useContext } from 'react';

// Create Auth Context
const AuthContext = createContext();

// Auth Provider Component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Login function
  const login = (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
    localStorage.setItem('brewai_user', JSON.stringify(userData));
    localStorage.setItem('brewai_authenticated', 'true');
  };

  // Register function
  const register = (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
    localStorage.setItem('brewai_user', JSON.stringify(userData));
    localStorage.setItem('brewai_authenticated', 'true');
  };

  // Logout function
  const logout = () => {
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('brewai_user');
    localStorage.removeItem('brewai_authenticated');
  };

  // Check if user is authenticated on app start
  React.useEffect(() => {
    const storedAuth = localStorage.getItem('brewai_authenticated');
    const storedUser = localStorage.getItem('brewai_user');
    
    if (storedAuth === 'true' && storedUser) {
      setUser(JSON.parse(storedUser));
      setIsAuthenticated(true);
    }
  }, []);

  return (
    <AuthContext.Provider value={{
      user,
      isAuthenticated,
      login,
      register,
      logout
    }}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};