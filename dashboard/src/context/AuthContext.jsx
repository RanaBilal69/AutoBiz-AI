import React, { createContext, useState, useContext, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { jwtDecode } from 'jwt-decode'

const AuthContext = createContext()

export const AuthProvider = ({ children }) => {
  const [user, setUser]   = useState(null)
  const [token, setToken] = useState(localStorage.getItem('token'))
  const navigate          = useNavigate()

  useEffect(() => {
    if (token) {
      try {
        const decoded = jwtDecode(token)
        setUser({ email: decoded.sub, role: decoded.role })
      } catch {
        logout()
      }
    }
  }, [token])

  // Listen for 401 events fired by axios interceptor (expired/invalid token on
  // protected endpoints). Uses a CustomEvent so the interceptor never does a
  // hard window.location reload — the SPA stays mounted and React Router handles
  // the navigation, avoiding Vercel 404s on direct-URL access.
  useEffect(() => {
    const handle = () => {
      logout()
      navigate('/login', { replace: true })
    }
    window.addEventListener('autobiz:unauthorized', handle)
    return () => window.removeEventListener('autobiz:unauthorized', handle)
  }, [navigate])

  const login = (accessToken) => {
    localStorage.setItem('token', accessToken)
    setToken(accessToken)
    const decoded = jwtDecode(accessToken)
    setUser({ email: decoded.sub, role: decoded.role })
  }

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('role')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)