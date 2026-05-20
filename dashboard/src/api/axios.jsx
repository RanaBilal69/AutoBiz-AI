import axios from 'axios';

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
});

// Attach token to every request
API.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 globally — but NEVER on the /auth/login endpoint itself.
// A wrong password returns 401 from /auth/login; intercepting that causes
// a hard page reload (window.location.href) which triggers a 404 on Vercel
// because Vercel doesn't know how to serve /login as a file.
// Only redirect when a protected endpoint returns 401 (expired/missing token).
API.interceptors.response.use(
  (response) => response,
  (error) => {
    const isAuthEndpoint = error.config?.url?.includes('/auth/');
    if (error.response?.status === 401 && !isAuthEndpoint) {
      localStorage.removeItem('token');
      localStorage.removeItem('role');
      // Use history-based navigation instead of hard reload to stay within the SPA
      window.dispatchEvent(new CustomEvent('autobiz:unauthorized'));
    }
    return Promise.reject(error);
  }
);

export default API;