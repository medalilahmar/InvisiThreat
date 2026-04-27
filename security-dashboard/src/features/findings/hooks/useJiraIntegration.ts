import { useState, useCallback } from 'react';
import { jiraApi } from '../../../api/services/jira';
import { JiraCreateIssueRequest, JiraIntegrationState, JiraErrorCode } from '../../../types/jira';
import type { Finding } from '../../../types/finding';

export function useJiraIntegration() {
  const [state, setState] = useState<JiraIntegrationState>({
    loading: false,
    error: null,
    success: false,
    jiraKey: null,
    jiraUrl: null,
  });

  const mapErrorCode = (status?: number, code?: string): JiraErrorCode => {
    if (status === 404) return 'NOT_FOUND';
    if (status === 409) return 'CONFLICT';
    if (status === 502) return 'SERVER_ERROR';
    if (status === 503) return 'SERVICE_UNAVAILABLE';
    if (status === 504) return 'GATEWAY_TIMEOUT';
    if (code === 'ECONNABORTED') return 'NETWORK_ERROR';
    return 'UNKNOWN';
  };

  const getErrorMessage = (errorCode: JiraErrorCode): string => {
    const messages: Record<JiraErrorCode, string> = {
      NOT_FOUND: ' Finding introuvable dans DefectDojo',
      CONFLICT: ' Un ticket Jira existe déjà pour ce finding',
      SERVER_ERROR: ' Backend InvisiThreat indisponible',
      SERVICE_UNAVAILABLE: ' Service Jira indisponible — vérifiez la configuration',
      GATEWAY_TIMEOUT: '⏱ Délai dépassé — la création a peut-être fonctionné, réessayez',
      NETWORK_ERROR: '⏱ Délai réseau dépassé — vérifiez votre connexion',
      UNKNOWN: 'Erreur lors de la création du ticket Jira',
    };
    return messages[errorCode];
  };

  const createJiraIssue = useCallback(
    async (finding: Finding, aiScore?: any) => {
      setState({
        loading: true,
        error: null,
        success: false,
        jiraKey: null,
        jiraUrl: null,
      });

      try {
        const payload: JiraCreateIssueRequest = {
          finding_id: finding.id,
          title: finding.title,
          severity: finding.severity.toUpperCase(),
          cvss_score: finding.cvss_score || 0,
          description: finding.description || '',
          cve: (finding as any).cve || '',
          cwe: (finding as any).cwe || '',
          file_path: finding.file_path || '',
          line: finding.line ?? undefined,
          tags: finding.tags || [],
          risk_level: aiScore?.risk_level || 'UNKNOWN',
          ai_score: aiScore?.risk_class,
          ai_confidence: aiScore?.confidence,
          engagement_id: finding.engagement_id,
          product_name: finding.product_name,
        };

        console.debug('[Jira] Creating issue with payload:', payload);
        // ✅ Correction : createIssue retourne directement JiraIssueResponse
        const response = await jiraApi.createIssue(payload);
        const jiraKey = response.key;
        const jiraUrl = jiraApi.getIssueUrl(jiraKey);

        console.info(`[Jira] Issue created successfully: ${jiraKey}`);

        setState({
          loading: false,
          error: null,
          success: true,
          jiraKey,
          jiraUrl,
        });

        return { jiraKey, jiraUrl };
      } catch (error: any) {
        const status = error?.response?.status;
        const errorData = error?.response?.data;
        const errorCode = mapErrorCode(status, error?.code);
        const errorMessage = errorData?.detail || getErrorMessage(errorCode);

        console.error('[Jira] Error creating issue:', {
          status,
          errorCode,
          message: errorMessage,
          details: errorData,
        });

        setState({
          loading: false,
          error: errorMessage,
          success: false,
          jiraKey: null,
          jiraUrl: null,
        });

        throw error;
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState({
      loading: false,
      error: null,
      success: false,
      jiraKey: null,
      jiraUrl: null,
    });
  }, []);

  return {
    ...state,
    createJiraIssue,
    reset,
  };
}