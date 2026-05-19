"""
data_loader.py — Chargement et indexation des findings depuis le CSV local.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from server.utils import _parse_tags

logger = logging.getLogger("invisithreat.data_loader")

SEVERITY_MAP_NUM = {0: "info", 1: "low", 2: "medium", 3: "high", 4: "critical"}


class LocalDataLoader:

    def __init__(self, csv_path: Path):
        self.csv_path        = csv_path
        self.df_findings:    Optional[pd.DataFrame] = None
        self.products:       Dict[int, Dict]         = {}
        self.engagements:    Dict[int, Dict]         = {}
        self.tests:          Dict[int, Dict]         = {}
        self.findings_by_id: Dict[int, Dict]         = {}
        self._loaded_at:     Optional[datetime]      = None
        self._ready                                  = False

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        if not self.csv_path.exists():
            logger.warning(f"CSV introuvable : {self.csv_path}")
            return False
        try:
            self.df_findings = pd.read_csv(self.csv_path, low_memory=False)
            
            logger.info(
                f"[LocalDataLoader] CSV chargé : "
                f"{len(self.df_findings)} findings, "
                f"{len(self.df_findings.columns)} colonnes"
            )
            self._build_hierarchy()
            self._loaded_at = datetime.now(timezone.utc)
            self._ready     = True
            logger.info(
                f"[LocalDataLoader] Hiérarchie construite : "
                f"{len(self.products)} produits, "
                f"{len(self.engagements)} engagements"
            )
            return True
        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur de chargement : {e}")
            return False

    def get_findings_for_product(self, product_id: int) -> List[Dict]:
        if not self._ready or product_id not in self.products:
            return []
        results = []
        for eng_id in self.products[product_id].get("engagements", set()):
            if eng_id in self.engagements:
                for fid in self.engagements[eng_id].get("findings", []):
                    if fid in self.findings_by_id:
                        results.append(self.findings_by_id[fid])
        return results

    def get_findings_for_engagement(self, engagement_id: int) -> List[Dict]:
        if not self._ready or engagement_id not in self.engagements:
            return []
        return [
            self.findings_by_id[fid]
            for fid in self.engagements[engagement_id].get("findings", [])
            if fid in self.findings_by_id
        ]

    def get_all_findings(self) -> List[Dict]:
        return list(self.findings_by_id.values()) if self._ready else []

    def get_products(self) -> List[Dict]:
        return [
            {
                "id":               p["id"],
                "name":             p["name"],
                "description":      p.get("description", None),  # ← AJOUTE
                "created":          p.get("created",     None),  # ← AJOUTE
                "engagement_count": len(p.get("engagements", [])),
            }
            for p in self.products.values()
        ]

    def get_engagements_for_product(self, product_id: int) -> List[Dict]:
        if product_id not in self.products:
            return []
        results = []
        for eng_id in self.products[product_id].get("engagements", set()):
            if eng_id in self.engagements:
                eng = self.engagements[eng_id]
                results.append({
                    "id":         eng["id"],
                    "name":       eng["name"],
                    "product_id": product_id,
                })
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_hierarchy(self) -> None:
        self.products.clear()
        self.engagements.clear()
        self.tests.clear()
        self.findings_by_id.clear()

        if self.df_findings is None:
            return

        try:
            required_cols = [
                "id", "product_id", "engagement_id",
                "product_name", "engagement_name", "severity_num",
            ]
            missing_cols = [c for c in required_cols if c not in self.df_findings.columns]
            if missing_cols:
                raise KeyError(f"Missing columns: {missing_cols}")

            # Produits
            for _, row in (
                self.df_findings[["product_id", "product_name", "created"]].drop_duplicates().iterrows()
            ):
                prod_id = int(row["product_id"])
                self.products[prod_id] = {
                    "id":          prod_id,
                    "name":        row["product_name"],
                    "engagements": set(),
                    "description": None,
                    "created":     row.get("created", None),

                }

            # Engagements
            for _, row in (
                self.df_findings[
                    ["product_id", "engagement_id", "engagement_name"]
                ].drop_duplicates().iterrows()
            ):
                eng_id  = int(row["engagement_id"])
                prod_id = int(row["product_id"])
                self.engagements[eng_id] = {
                    "id":         eng_id,
                    "name":       row["engagement_name"],
                    "product_id": prod_id,
                    "tests":      set(),
                }
                if prod_id in self.products:
                    self.products[prod_id]["engagements"].add(eng_id)

            # Findings
            for _, row in self.df_findings.iterrows():
                finding_id = int(row["id"])
                eng_id     = int(row["engagement_id"])
                prod_id    = int(row["product_id"])

                finding_dict               = row.to_dict()
                finding_dict["id"]         = finding_id
                finding_dict["product_id"] = prod_id
                finding_dict["engagement_id"] = eng_id

                # Sévérité
                sev_num = row["severity_num"]
                if pd.isna(sev_num):
                    severity = "info"
                else:
                    try:
                        severity = SEVERITY_MAP_NUM.get(int(sev_num), "info")
                    except (ValueError, TypeError):
                        severity = "info"
                finding_dict["severity"] = severity

                # Tags
                tags_list = []
                if row.get("tag_urgent")        == 1: tags_list.append("urgent")
                if row.get("tag_in_production") == 1: tags_list.append("production")
                if row.get("tag_sensitive")     == 1: tags_list.append("sensitive")
                if row.get("tag_external")      == 1: tags_list.append("external")
                finding_dict["tags"] = tags_list

                self.findings_by_id[finding_id] = finding_dict

                if eng_id in self.engagements:
                    self.engagements[eng_id].setdefault("findings", []).append(finding_id)

        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur _build_hierarchy : {e}")
            self._ready = False
            raise