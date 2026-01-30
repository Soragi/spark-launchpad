import { useState, useEffect, useCallback } from 'react';
import { Launchable } from '@/components/launchables/LaunchableCard';

const STORAGE_KEY = 'saved-launchables';

export function useSavedLaunchables() {
  const [savedLaunchables, setSavedLaunchables] = useState<Launchable[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setSavedLaunchables(JSON.parse(stored));
      } catch {
        console.error('Failed to parse saved launchables');
      }
    }
  }, []);

  // Save to localStorage whenever savedLaunchables changes
  const persistToStorage = useCallback((launchables: Launchable[]) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(launchables));
  }, []);

  const addLaunchable = useCallback((launchable: Launchable) => {
    setSavedLaunchables((prev) => {
      // Don't add if already exists
      if (prev.some((l) => l.id === launchable.id)) {
        return prev;
      }
      const updated = [...prev, launchable];
      persistToStorage(updated);
      return updated;
    });
  }, [persistToStorage]);

  const removeLaunchable = useCallback((launchableId: string) => {
    setSavedLaunchables((prev) => {
      const updated = prev.filter((l) => l.id !== launchableId);
      persistToStorage(updated);
      return updated;
    });
  }, [persistToStorage]);

  const isLaunchableSaved = useCallback((launchableId: string) => {
    return savedLaunchables.some((l) => l.id === launchableId);
  }, [savedLaunchables]);

  return {
    savedLaunchables,
    addLaunchable,
    removeLaunchable,
    isLaunchableSaved,
  };
}
