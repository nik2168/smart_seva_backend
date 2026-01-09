"""
ela_processor.py

Improved Error Level Analysis (ELA) processor:
 - Multi-Quality ELA (MQ-ELA)
 - DCT-based comparison when input is JPEG (via jpegio)
 - CLAHE enhancement + high-frequency residual fusion
 - Better adaptive thresholding and scoring
 - Heatmap export and debug metrics

Returns:
{
 "module": "ELA",
 "score": float(0-100),
 "details": { ... debug metrics ... },
 "status": "success" / "error"
}
"""

import os
import cv2
import numpy as np
from skimage import img_as_ubyte
import jpegio as jio  # optional, used when input is JPEG
from typing import Tuple, Dict, Any

class ELAProcessor:
    def __init__(
        self,
        qualities=(75, 85, 95),
        multiplier: float = 15.0,
        min_region_pct: float = 0.0005,  # 0.05% of image
        use_dct_when_possible: bool = True,
        save_heatmap: bool = True
    ):
        """
        :param qualities: JPEG qualities to use for MQ-ELA (list/tuple).
        :param multiplier: brightness multiplier for ELA visualization.
        :param min_region_pct: minimum region area (fraction of image) to consider suspicious.
        :param use_dct_when_possible: if True, attempt DCT-level processing using jpegio for JPEG inputs.
        :param save_heatmap: if True, save heatmap next to original file.
        """
        self.qualities = list(qualities)
        self.multiplier = multiplier
        self.min_region_pct = min_region_pct
        self.use_dct_when_possible = use_dct_when_possible
        self.save_heatmap = save_heatmap

    # -------------------------
    # Public API
    # -------------------------
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Main entry point. Returns structured result with score and debug details.
        """
        try:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(image_path)

            # Read with OpenCV (keeps pixel array in BGR)
            original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if original is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Normalize to 3-channel BGR uint8
            original = self._ensure_bgr_uint8(original)

            # If input is JPEG AND jpegio is available & enabled -> attempt DCT route
            ext = os.path.splitext(image_path)[1].lower()
            use_dct = self.use_dct_when_possible and ext in ('.jpg', '.jpeg')
            if use_dct:
                try:
                    score, metrics, heatmap = self._analyze_with_dct(image_path, original)
                except Exception:
                    # Fall back to pixel MQ-ELA if dct fails for any reason
                    score, metrics, heatmap = self._analyze_mqela(original, image_path)
            else:
                score, metrics, heatmap = self._analyze_mqela(original, image_path)

            result = {
                "module": "ELA",
                "score": round(float(score), 2),
                "details": {
                    **metrics,
                    "heatmap_path": heatmap if self.save_heatmap else None
                },
                "status": "success"
            }
            return result

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            return {
                "module": "ELA",
                "score": 0.0,
                "status": "error",
                "error_msg": error_details
            }

    # -------------------------
    # Helpers
    # -------------------------
    def _ensure_bgr_uint8(self, img: np.ndarray) -> np.ndarray:
        # Convert RGBA -> BGR, grayscale -> BGR, float -> uint8
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)
        
        # Check dimensions properly - ensure we check ndim before accessing shape[2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            # Safely check number of channels
            num_channels = img.shape[2] if len(img.shape) > 2 else 1
            if num_channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # If already 3-channel BGR, leave it as is
        
        return img

    # -------------------------
    # DCT-based (jpegio) path
    # -------------------------
    def _analyze_with_dct(self, image_path: str, original_bgr: np.ndarray) -> Tuple[float, Dict[str, Any], str]:
        """
        Uses jpegio to obtain DCT coefficients and quantization tables.
        Idea: compute differences between original DCT coefficients and recompressed DCTs
        to better localize recompression anomalies (stronger than pixel-level ELA).
        """
        # Read jpeg file
        jpeg = jio.read(image_path)
        # jpeg.coef_arrays is a list of Y, Cb, Cr DCT arrays
        dct_arrays = jpeg.coef_arrays  # list of 2D arrays per channel
        # We'll compute a simple heuristic: local variance of DCT blocks
        # Convert original BGR to YCrCb for block-wise mapping
        ycc = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2YCrCb)
        gray_y = ycc[:, :, 0]

        # Build a block-map of DCT energy using absolute sum of coefficients in each 8x8 block
        block_map = self._dct_energy_map(dct_arrays, gray_y.shape)

        # Normalize and convert to heatmap
        norm = cv2.normalize(block_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Enhance via CLAHE + residual fusion (similar to pixel path)
        enhanced = self._enhance_gray(norm)
        final_map = self._fuse_with_residual(enhanced)

        # Threshold and get blobs
        score, metrics = self._score_from_map(final_map)

        # Save heatmap
        heatmap_path = None
        if self.save_heatmap:
            heatmap_path = self._save_heatmap_color(final_map, image_path, suffix="_ela_dct")

        return score, metrics, heatmap_path

    def _dct_energy_map(self, dct_arrays, target_shape):
        """
        Build a per-block energy map (upsampled to image size).
        dct_arrays: typically [Y_coef_array, Cb_coef_array, Cr_coef_array]
        """
        # Use Y channel DCT if available - check if array exists and is not empty
        if not dct_arrays or len(dct_arrays) == 0:
            raise ValueError("dct_arrays is empty")
        
        y_dct = dct_arrays[0]
        if y_dct is None or y_dct.size == 0:
            raise ValueError("Y channel DCT array is empty")
        
        # y_dct shape is (h_blocks, w_blocks, 8, 8) or sometimes (h_blocks*8, w_blocks*8) depending on jpegio version.
        # jpegio returns coef_arrays with shape (h_blocks*8, w_blocks*8) flattened; we attempt robust handling.
        if y_dct.ndim == 2:
            # If it's flattened pixel-like, compute local 8x8 energy with block processing
            h, w = y_dct.shape
            # compute energy via moving 8x8 block sum of abs(dct)
            energy = self._block_energy_from_flat(y_dct)
        elif y_dct.ndim >= 4:
            # If coef arrays are block-arrays (4D: h_blocks, w_blocks, 8, 8)
            # compute absolute-sum per-block
            block_energy = np.sum(np.abs(y_dct), axis=tuple(range(2, y_dct.ndim)))
            # Upsample block map to image size
            h_blocks, w_blocks = block_energy.shape
            energy = cv2.resize(block_energy.astype(np.float32), (w_blocks * 8, h_blocks * 8),
                                interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback: treat as 2D and use block energy
            energy = self._block_energy_from_flat(y_dct)
        
        # Resize energy to target image size (gray shape)
        target_h, target_w = target_shape
        energy = cv2.resize(energy, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return energy

    def _block_energy_from_flat(self, flat):
        # Sum absolute values in non-overlapping 8x8 blocks for a flattened-like DCT array
        h, w = flat.shape
        bh = h // 8
        bw = w // 8
        energy = np.zeros((bh, bw), dtype=np.float32)
        for by in range(bh):
            for bx in range(bw):
                block = flat[by*8:(by+1)*8, bx*8:(bx+1)*8]
                energy[by, bx] = np.sum(np.abs(block))
        energy_up = cv2.resize(energy, (bw*8, bh*8), interpolation=cv2.INTER_NEAREST)
        return energy_up

    # -------------------------
    # MQ-ELA and pixel path
    # -------------------------
    def _analyze_mqela(self, original_bgr: np.ndarray, image_path: str) -> Tuple[float, Dict[str, Any], str]:
        """
        Multi-quality ELA: resave at several JPEG qualities, compute absdiff across each,
        average them, enhance, then compute score.
        """
        h, w = original_bgr.shape[:2]
        elas = []
        for q in self.qualities:
            # re-encode to jpeg in memory
            success, buf = cv2.imencode('.jpg', original_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
            if not success:
                continue
            recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            diff = cv2.absdiff(original_bgr, recompressed).astype(np.float32)
            elas.append(diff)

        if not elas:
            raise RuntimeError("MQ-ELA failed to produce any recompressed images.")

        # average ELA across qualities (mean of absolute diffs per channel)
        avg_ela = np.mean(elas, axis=0).astype(np.uint8)
        # Convert to grayscale (luminance)
        gray = cv2.cvtColor(avg_ela, cv2.COLOR_BGR2GRAY)
        # Amplify
        gray = cv2.convertScaleAbs(gray, alpha=self.multiplier)

        # CLAHE enhancement
        enhanced = self._enhance_gray(gray)

        # High-frequency residual fusion
        final_map = self._fuse_with_residual(enhanced)

        # Threshold and score
        score, metrics = self._score_from_map(final_map)

        # Save heatmap
        heatmap_path = None
        if self.save_heatmap and image_path:
            heatmap_path = self._save_heatmap_color(final_map, image_path, suffix="_ela_mq")

        return score, metrics, heatmap_path

    def _enhance_gray(self, gray: np.ndarray) -> np.ndarray:
        """
        CLAHE to improve local contrast and reveal subtle artifacts.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _fuse_with_residual(self, gray: np.ndarray) -> np.ndarray:
        """
        Fuse gray ELA map with a high-frequency residual (Laplacian) to increase sensitivity.
        """
        # Laplacian/high-pass kernel
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = cv2.convertScaleAbs(lap)
        # Weighted fusion (0.7 * ELA + 0.3 * residual)
        fused = cv2.addWeighted(gray.astype(np.float32), 0.7, lap.astype(np.float32), 0.3, 0)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return fused

    def _score_from_map(self, final_map: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Compute score from final map:
         - Use percentile + Otsu to adapt threshold
         - Morphology to connect regions
         - Compute area and intensity statistics
        """
        # Stats
        mean_val = float(np.mean(final_map))
        std_val = float(np.std(final_map))
        max_val = int(np.max(final_map))
        p95 = float(np.percentile(final_map, 95))
        p99 = float(np.percentile(final_map, 99))

        # Combine Otsu and percentile for robust threshold
        # Ensure otsu_thresh is a scalar (cv2.threshold returns scalar, but be safe)
        _, otsu_thresh = cv2.threshold(final_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert to scalar safely - handle both scalar and array returns
        if isinstance(otsu_thresh, np.ndarray):
            if otsu_thresh.size == 1:
                otsu_thresh = float(otsu_thresh.item())
            else:
                otsu_thresh = float(otsu_thresh.flat[0])  # Take first element if array
        else:
            otsu_thresh = float(otsu_thresh)
        # Ensure all values are scalars before using max()
        thresh_val = max(otsu_thresh, p95, mean_val + 1.5 * std_val)
        _, th = cv2.threshold(final_map, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphology: close small holes and connect
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = final_map.shape
        total_pixels = h * w
        min_area = max(1, int(self.min_region_pct * total_pixels))

        suspicious_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            mask = np.zeros(final_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            # Get region pixels safely - check if mask has any non-zero pixels
            region_pixels = final_map[mask > 0]
            if region_pixels.size == 0:
                continue  # Skip if no pixels in region
            region_intensity = float(np.mean(region_pixels))
            suspicious_regions.append({"area": area, "intensity": region_intensity})

        if not suspicious_regions:
            metrics = {
                "max_diff": max_val,
                "avg_diff": mean_val,
                "suspicious_blobs": 0,
                "largest_tamper_area": 0,
                "total_tamper_area": 0,
                "relative_size_percent": 0.0,
                "debug_threshold": float(thresh_val)
            }
            return 0.0, metrics

        total_tamper_area = sum(r["area"] for r in suspicious_regions)
        max_blob_area = max(r["area"] for r in suspicious_regions)
        max_intensity = max(r["intensity"] for r in suspicious_regions)
        avg_intensity = np.mean([r["intensity"] for r in suspicious_regions])

        relative_size = (max_blob_area / total_pixels) * 100.0
        total_relative_size = (total_tamper_area / total_pixels) * 100.0

        # Conservative area scoring (log-like)
        if relative_size < 0.05:
            area_score = relative_size * 40
        elif relative_size < 0.2:
            area_score = 2 + (relative_size - 0.05) * 20
        elif relative_size < 0.5:
            area_score = 5 + (relative_size - 0.2) * 30
        elif relative_size < 1.0:
            area_score = 14 + (relative_size - 0.5) * 35
        elif relative_size < 2.0:
            area_score = 32 + (relative_size - 1.0) * 20
        elif relative_size < 5.0:
            area_score = 52 + (relative_size - 2.0) * 12
        else:
            area_score = 88 + min(12, (relative_size - 5.0) * 2)

        # Intensity score
        intensity_score = 0.0
        # relative to threshold; avoid division by zero
        denom = max(1.0, float(thresh_val))
        max_intensity_ratio = max_intensity / denom
        avg_intensity_ratio = avg_intensity / denom

        if max_intensity_ratio > 2.0:
            intensity_score = min(30.0, (max_intensity_ratio - 2.0) * 10.0)
        elif max_intensity_ratio > 1.5:
            intensity_score = min(15.0, (max_intensity_ratio - 1.5) * 10.0)

        if avg_intensity_ratio > 1.5:
            intensity_score += min(20.0, (avg_intensity_ratio - 1.5) * 10.0)

        base_score = area_score + 0.4 * intensity_score

        # Consider total area multiplier only when substantial
        num_regions = len(suspicious_regions)
        if total_relative_size > 1.0 and num_regions > 3:
            area_ratio = total_relative_size / max(relative_size, 0.1)
            if area_ratio > 2.0:
                multiplier = 1.0 + min(0.5, (total_relative_size - 1.0) * 0.1)
                base_score = min(100.0, base_score * multiplier)
            elif area_ratio > 1.5 and num_regions > 10:
                multiplier = 1.0 + min(0.3, (num_regions - 10) * 0.02)
                base_score = min(100.0, base_score * multiplier)

        score = min(100.0, base_score)
        metrics = {
            "max_diff": int(max_val),
            "avg_diff": float(mean_val),
            "suspicious_blobs": int(num_regions),
            "largest_tamper_area": float(max_blob_area),
            "total_tamper_area": float(total_tamper_area),
            "relative_size_percent": round(float(relative_size), 4),
            "total_relative_size_percent": round(float(total_relative_size), 4),
            "max_intensity_ratio": round(float(max_intensity_ratio), 3),
            "avg_intensity_ratio": round(float(avg_intensity_ratio), 3),
            "area_score": round(float(area_score), 3),
            "intensity_score": round(float(intensity_score), 3),
            "base_score": round(float(base_score), 3),
            "debug_threshold": float(thresh_val)
        }
        return float(score), metrics

    def _save_heatmap_color(self, final_map: np.ndarray, original_path: str, suffix: str = "_ela"):
        """
        Convert grayscale final_map to jet colormap and save next to original file.
        """
        color = cv2.applyColorMap(final_map, cv2.COLORMAP_JET)
        base, ext = os.path.splitext(original_path)
        out = f"{base}{suffix}{ext}"
        cv2.imwrite(out, color)
        return out


# Example usage
if __name__ == "__main__":
    processor = ELAProcessor()
    res = processor.analyze("src/test_data/bright.jpeg")
    print(res)


    # output example:
    # {
    #     "module": "ELA",
    #     "score": 100.0,
    #     "details": {
    #         "max_diff": 255,
    #         "avg_diff": 127.5,
    #         "suspicious_blobs": 1,
    #         "largest_tamper_area": 100.0,
    #         "total_tamper_area": 100.0,
    #         "relative_size_percent": 0.05,
    #         "total_relative_size_percent": 0.05,
    #         "max_intensity_ratio": 2.0,
    #         "avg_intensity_ratio": 1.5,
    #         "area_score": 10.0,
    #         "intensity_score": 10.0,
    #         "base_score": 20.0,
    #         "debug_threshold": 127.5
    #     },
    #     "status": "success"
    # }