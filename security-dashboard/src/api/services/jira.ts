import { apiClient } from '../client';
import type { 
  JiraCreateIssueRequest, 
  JiraHealthResponse 
} from '../../types/jira';

export interface JiraCheckIssueResponse {
  exists: boolean;
  jira_key: string | null;
  jira_url: string | null;
  created?: string;
}

export interface JiraIssueResponse {
  key: string;
  url?: string;
}

export const jiraApi = {
  /**
   * Crée un ticket Jira à partir d'un finding
   * @param payload Les données du finding (incluant finding_id)
   * @returns La réponse Jira contenant la clé et l'URL du ticket
   */
  async createIssue(payload: JiraCreateIssueRequest): Promise<JiraIssueResponse> {
    const findingId = payload.finding_id;
    const res = await apiClient.post<JiraIssueResponse>(
      `/defectdojo/findings/${findingId}/create-jira-issue`,
      payload
    );
    return res.data;
  },

  /**
   * Vérifie l'existence d'un ticket Jira pour un finding donné
   * @param findingId ID du finding
   * @returns Informations sur le ticket existant ou inexistant
   */
  async checkIssue(findingId: number): Promise<JiraCheckIssueResponse> {
    try {
      const res = await apiClient.get<JiraCheckIssueResponse>(
        `/defectdojo/findings/${findingId}/jira-issue`
      );
      return res.data;
    } catch (error) {
      console.error('[Jira] Error checking issue:', error);
      return { 
        exists: false, 
        jira_key: null, 
        jira_url: null 
      };
    }
  },

  /**
   * Vérifie la santé de la connexion à Jira
   * @returns Statut de la connexion, informations du projet
   */
  health: () => apiClient.get<JiraHealthResponse>('/jira/health'),

  /**
   * Construit l'URL d'un ticket Jira à partir de sa clé
   * @param jiraKey Clé du ticket (ex: INV-123)
   * @returns URL complète vers le ticket
   */
  getIssueUrl: (jiraKey: string): string => {
    const jiraServer = import.meta.env.VITE_JIRA_SERVER || 'https://medalilahmar.atlassian.net';
    return `${jiraServer}/browse/${jiraKey}`;
  },
};