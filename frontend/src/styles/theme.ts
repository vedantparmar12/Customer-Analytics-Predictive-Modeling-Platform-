import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea',
      light: '#9F7AEA',
      dark: '#553C9A',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#764ba2',
      light: '#9F6DB3',
      dark: '#4C1E6E',
      contrastText: '#ffffff',
    },
    success: {
      main: '#27ae60',
      light: '#52c77e',
      dark: '#1e8449',
    },
    warning: {
      main: '#f39c12',
      light: '#f5b041',
      dark: '#d68910',
    },
    error: {
      main: '#e74c3c',
      light: '#ec7063',
      dark: '#c0392b',
    },
    info: {
      main: '#3498db',
      light: '#5dade2',
      dark: '#2874a6',
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    text: {
      primary: '#2c3e50',
      secondary: '#7f8c8d',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      color: '#2c3e50',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      color: '#2c3e50',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: '#2c3e50',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#2c3e50',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      color: '#2c3e50',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      color: '#2c3e50',
    },
    body1: {
      fontSize: '1rem',
      color: '#2c3e50',
    },
    body2: {
      fontSize: '0.875rem',
      color: '#7f8c8d',
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.05)',
    '0px 4px 8px rgba(0,0,0,0.08)',
    '0px 6px 12px rgba(0,0,0,0.1)',
    '0px 8px 16px rgba(0,0,0,0.12)',
    '0px 10px 20px rgba(0,0,0,0.15)',
    '0px 12px 24px rgba(0,0,0,0.18)',
    '0px 14px 28px rgba(0,0,0,0.2)',
    '0px 16px 32px rgba(0,0,0,0.22)',
    '0px 18px 36px rgba(0,0,0,0.25)',
    '0px 20px 40px rgba(0,0,0,0.28)',
    '0px 22px 44px rgba(0,0,0,0.3)',
    '0px 24px 48px rgba(0,0,0,0.32)',
    '0px 26px 52px rgba(0,0,0,0.35)',
    '0px 28px 56px rgba(0,0,0,0.38)',
    '0px 30px 60px rgba(0,0,0,0.4)',
    '0px 32px 64px rgba(0,0,0,0.42)',
    '0px 34px 68px rgba(0,0,0,0.45)',
    '0px 36px 72px rgba(0,0,0,0.48)',
    '0px 38px 76px rgba(0,0,0,0.5)',
    '0px 40px 80px rgba(0,0,0,0.52)',
    '0px 42px 84px rgba(0,0,0,0.55)',
    '0px 44px 88px rgba(0,0,0,0.58)',
    '0px 46px 92px rgba(0,0,0,0.6)',
    '0px 48px 96px rgba(0,0,0,0.62)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
          padding: '8px 24px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 12px rgba(0,0,0,0.15)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #5a67d8 0%, #6b4199 100%)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            '& fieldset': {
              borderColor: '#e0e0e0',
              borderWidth: 2,
            },
            '&:hover fieldset': {
              borderColor: '#667eea',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#667eea',
            },
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#f8f9fa',
          borderRight: '1px solid #e0e0e0',
        },
      },
    },
  },
});