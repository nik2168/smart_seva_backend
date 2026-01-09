import exiftool
import json
import os
from typing import Dict, Any

class MetadataProcessor:
    """
    Module 4: Metadata Forensics
    Technology: ExifTool (via PyExifTool)
    Purpose: Extract hidden XMP/IPTC data to find the 'Software' signature.
    """

    def __init__(self, exiftool_path: str = None):
        # If exiftool is not in system PATH, specify it here
        self.exiftool_path = exiftool_path

    def analyze(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"status": "error", "message": "File not found"}

        try:
            # Use PyExifTool to get ALL metadata
            with exiftool.ExifToolHelper(executable=self.exiftool_path) as et:
                # -G flag groups output (e.g., 'EXIF:Model', 'XMP:Creator')
                metadata_list = et.get_metadata(image_path)
                
            if not metadata_list:
                return {"score": 0, "status": "error", "msg": "No metadata found"}

            # ExifTool returns a list (one per file), we just want the first one
            meta = metadata_list[0]
            
            # --- EXTRACT CRITICAL FIELDS ---
            # We look for specific keys that reveal editing history
            
            # 1. Software / Creator Tool (The Smoking Gun)
            software = (
                meta.get("EXIF:Software") or 
                meta.get("XMP:CreatorTool") or 
                meta.get("XMP:HistorySoftwareAgent") or
                "Unknown"
            )

            # 2. Author / Artist
            artist = (
                meta.get("EXIF:Artist") or 
                meta.get("XMP:Creator") or 
                meta.get("IPTC:By-line") or
                "Unknown"
            )

            # 3. Modification Dates
            create_date = meta.get("EXIF:CreateDate") or meta.get("XMP:CreateDate")
            modify_date = meta.get("EXIF:ModifyDate") or meta.get("File:FileModifyDate")

            # 4. Technical Details (Deep Forensics)
            compression = meta.get("File:FileType")  # e.g., JPEG, PNG
            subsampling = meta.get("File:YCbCrSubSampling") # e.g., 'YCbCr4:2:0 (2 2)'

            # --- FORENSIC SCORING LOGIC ---
            score = 0
            flags = []

            # CHECK 1: Suspicious Software (Canva, Photoshop, GIMP)
            suspicious_tools = ["Canva", "Photoshop", "GIMP", "Paint", "Editor"]
            
            if any(tool.lower() in str(software).lower() for tool in suspicious_tools):
                score += 80
                flags.append(f"Edited with: {software}")

            # CHECK 2: Metadata Stripped? (Common in social media downloads)
            if software == "Unknown" and artist == "Unknown" and "EXIF:Make" not in meta:
                # Not necessarily fake, but means origin is lost
                score += 20
                flags.append("Metadata stripped/missing (Social Media or Screenshot?)")

            # CHECK 3: Creator Name Mismatch
            # If the certificate says "Govt of India" but metadata says "Mercy Nik"
            if artist != "Unknown":
                flags.append(f"Creator found in metadata: {artist}")
                # We add a moderate score because a scan shouldn't have an 'Artist' usually
                score += 30

            # Cap score
            score = min(100, score)

            return {
                "module": "MetadataForensics",
                "score": score,
                "status": "success",
                "details": {
                    "software_used": software,
                    "creator": artist,
                    "dates": {
                        "created": create_date,
                        "modified": modify_date
                    },
                    "technical": {
                        "subsampling": subsampling,
                        "file_type": compression,
                        "image_size": meta.get("File:ImageSize")
                    },
                    "forensic_flags": flags,
                    "full_dump_subset": {k:v for k,v in meta.items() if "Canva" in str(v) or "Mercy" in str(v)}
                }
            }

        except Exception as e:
            return {
                "module": "MetadataForensics",
                "score": 0,
                "status": "error",
                "error_msg": str(e)
            }

if __name__ == "__main__":
    # Example Usage
    processor = MetadataProcessor() # Add path to exiftool if needed
    result = processor.analyze("src/test_data/og.jpeg")
    print(json.dumps(result, indent=2))


# uv run python -m src.services.processors.image_forensics_processors.metadata_processor