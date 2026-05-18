import { configureStore } from '@reduxjs/toolkit';
import authReducer from './authSlice';
import { useTheme, useToggleTheme, useSetTheme } from './useThemeStore';


export const store = configureStore({
  reducer: {
    auth: authReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export { useTheme, useToggleTheme, useSetTheme };