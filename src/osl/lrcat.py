from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple, Any


# Notes on Lightroom Classic catalog schema (observed across versions 6-13+):
# - AgLibraryRootFolder(absolutePath)
# - AgLibraryFolder(rootFolder, pathFromRoot)
# - AgLibraryFile(folder, baseName, extension, id_local)
# - Adobe_images(rootFile -> AgLibraryFile.id_local, id_local)
# - AgHarvestedExifMetadata(image -> Adobe_images.id_local, cameraModelRef, lensRef,
#                           isoSpeedRating, exposureTime, aperture, focalLength, ...)
# - AgInternedExifCameraModel(id_local, value)
# - AgInternedExifLens(id_local, value)
#
# The schema can vary slightly by version/locale. All queries here are defensive.


@dataclass
class PhotoRecord:
    """Flattened photo metadata sufficient for ingestion manifest."""
    id: int
    raw_path: Path
    camera: Optional[str]
    lens: Optional[str]
    iso: Optional[int]
    shutter: Optional[str]  # stored as normalized string (e.g., "1/200" or "0.5")
    aperture: Optional[float]
    focal_length: Optional[float]
    # Optional session/grouping key (folder path by default)
    session_id: Optional[str]


def open_catalog_readonly(path: Path) -> sqlite3.Connection:
    """Open Lightroom catalog as read-only SQLite connection with row factory."""
    # Use URI to enforce read-only to avoid any accidental writes
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def iter_photo_records(conn: sqlite3.Connection, limit: Optional[int] = None) -> Iterator[PhotoRecord]:
    """Iterate photo records with defensive joins across common LR tables."""
    # Core query assembling absolute file path and basic EXIF-derived metadata
    # Some catalogs may miss certain tables/columns; we guard with try/except and alternative queries.
    sql = """
    SELECT
        f.id_local                               AS file_id,
        rf.absolutePath                          AS root_path,
        COALESCE(fol.pathFromRoot, '')           AS rel_path,
        f.baseName                               AS base_name,
        COALESCE(f.extension, '')                AS extension,
        cam.value                                AS camera,
        lens.value                               AS lens,
        exif.*
    FROM AgLibraryFile f
    JOIN AgLibraryFolder fol ON f.folder = fol.id_local
    JOIN AgLibraryRootFolder rf ON fol.rootFolder = rf.id_local
    LEFT JOIN Adobe_images img ON img.rootFile = f.id_local
    LEFT JOIN AgHarvestedExifMetadata exif ON exif.image = img.id_local
    LEFT JOIN AgInternedExifCameraModel cam ON cam.id_local = exif.cameraModelRef
    LEFT JOIN AgInternedExifLens lens ON lens.id_local = exif.lensRef
    ORDER BY f.id_local ASC
    """
    cur = conn.cursor()
    try:
        if limit is not None:
            sql_limited = sql + " LIMIT ?"
            cur.execute(sql_limited, (int(limit),))
        else:
            cur.execute(sql)
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to query Lightroom catalog: {e}")

    for row in cur:
        raw_path = _compose_path(
            root=row["root_path"],
            rel=row["rel_path"],
            base=row["base_name"],
            ext=row["extension"],
        )
        # Safely access EXIF fields by checking for key existence in the row dict
        # This handles schema variations where columns might be missing or named differently.
        row_keys = row.keys()
        iso = row["isoSpeedRating"] if "isoSpeedRating" in row_keys else None
        aperture = row["aperture"] if "aperture" in row_keys else None
        focal_length = row["focalLength"] if "focalLength" in row_keys else None
        # Handle alternative names for shutter speed
        exposure_time = row["exposureTime"] if "exposureTime" in row_keys else None
        if exposure_time is None:
            exposure_time = row["shutterSpeed"] if "shutterSpeed" in row_keys else None

        yield PhotoRecord(
            id=int(row["file_id"]),
            raw_path=raw_path,
            camera=_safe_str(row["camera"]),
            lens=_safe_str(row["lens"]),
            iso=_safe_int(iso),
            shutter=_normalize_exposure_time(exposure_time),
            aperture=_safe_float(aperture),
            focal_length=_safe_float(focal_length),
            session_id=_derive_session_id(row["rel_path"]),
        )


def count_photos(conn: sqlite3.Connection) -> int:
    """Return total number of files tracked by the catalog."""
    try:
        cur = conn.execute("SELECT COUNT(1) FROM AgLibraryFile")
        (n,) = cur.fetchone()
        return int(n)
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to count photos: {e}")


def _compose_path(root: Optional[str], rel: Optional[str], base: Optional[str], ext: Optional[str]) -> Path:
    # Lightroom typically stores root as absolute path with trailing separator
    # pathFromRoot is a relative path (may be empty string) that can contain subfolders and trailing slash
    root_path = Path(root or "")
    rel_path = Path((rel or "").lstrip("/\\"))
    base_name = f"{base}" if base else ""
    extension = ext or ""
    file_name = f"{base_name}.{extension}" if extension else base_name
    full = (root_path / rel_path / file_name).resolve()
    return full


def _safe_str(x: Any) -> Optional[str]:
    try:
        return str(x) if x is not None else None
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        try:
            return int(float(x)) if x is not None else None
        except Exception:
            return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _normalize_exposure_time(x: Any) -> Optional[str]:
    """Normalize exposure time into a user-friendly string.

    Input may be stored as a float (seconds) or as a string like '1/250'.
    We prefer:
        - fraction for t <= 1s (nearest integer denominator)
        - decimal with 1-3 decimals for t > 1s
    """
    if x is None:
        return None
    # If it's already a ratio-like string, pass through
    try:
        s = str(x)
        if "/" in s:
            return s
    except Exception:
        pass
    # Try numeric seconds
    secs = _safe_float(x)
    if secs is None:
        return None
    if secs <= 0:
        return None
    if secs > 1.0:
        # 1 to 999 seconds with precision
        if secs < 10:
            return f"{secs:.2f}s"
        elif secs < 100:
            return f"{secs:.1f}s"
        else:
            return f"{int(round(secs))}s"
    # Convert to nearest nice fraction denominator
    # Try common shutter denominators used by cameras
    denominators = [2, 3, 4, 5, 6, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 4000, 8000]
    best = None
    best_err = 1e9
    for d in denominators:
        err = abs(1.0 / d - secs)
        if err < best_err:
            best_err = err
            best = d
    if best is None:
        return f"{secs:.3f}s"
    return f"1/{best}"


def _derive_session_id(rel_path: Optional[str]) -> Optional[str]:
    """Heuristic 'session' id: top one or two directories of pathFromRoot, if present."""
    if not rel_path:
        return None
    # Normalize separators and strip trailing slash
    parts = [p for p in Path(rel_path).parts if p not in ("/", "\\")]
    if not parts:
        return None
    # Use the first directory as session; if only one level, that's fine.
    return parts[0]


__all__ = [
    "PhotoRecord",
    "open_catalog_readonly",
    "iter_photo_records",
    "count_photos",
]