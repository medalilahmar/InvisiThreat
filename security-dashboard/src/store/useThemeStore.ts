// src/store/useThemeStore.ts
import { create } from 'zustand';
import { createThemeSlice, ThemeState } from './themeSlice';

export const useThemeStore = create<ThemeState>()((set) => ({
  ...createThemeSlice(set),
}));

// Sélecteurs optimisés
export const useTheme = () => useThemeStore((state) => state.theme);
export const useToggleTheme = () => useThemeStore((state) => state.toggleTheme);
export const useSetTheme = () => useThemeStore((state) => state.setTheme);