import type { LoginCredentials, AuthUser } from '../types/auth';



const MOCK_USERS: Record<string, { password: string; role: AuthUser['role'] }> = {
  admin: { password: 'admin', role: 'admin' },
};

export const authService = {
  async login(credentials: LoginCredentials): Promise<AuthUser> {
    // Simulate network delay
    await new Promise((r) => setTimeout(r, 700));

    const found = MOCK_USERS[credentials.username.toLowerCase()];
    if (!found || found.password !== credentials.password) {
      throw new Error('Invalid username or password');
    }

    return { username: credentials.username.toLowerCase(), role: found.role };
  },
};