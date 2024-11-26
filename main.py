# semantic_parser.py

import pdfplumber
import fitz
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import spacy
import re
import logging
from pathlib import Path
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
from collections import Counter

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

@dataclass
class PageElement:
    content: Any
    element_type: str
    confidence: float
    bounding_box: BoundingBox
    metadata: Dict[str, Any]

class PDFSemanticParser:
    """Enhanced PDF parser with semantic understanding and structured data extraction"""

    def __init__(self, spacy_model="en_core_web_sm"):
        """Initialize the parser with required models and configurations"""
        self.nlp = spacy.load(spacy_model)
        self.logger = self._setup_logging()

        # Configure element detection patterns
        self.patterns = {
            'table_headers': r'^[A-Z\s]+$',
            'amounts': r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'section_headers': r'^[A-Z][A-Za-z\s]+:',
        }

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the parser"""
        logger = logging.getLogger('PDFSemanticParser')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def parse_document(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Main method to parse PDF document with semantic understanding

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for extracted content

        Returns:
            Dictionary containing structured document content
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            with pdfplumber.open(pdf_path) as pdf:
                document_structure = {
                    'metadata': self._extract_metadata(pdf),
                    'pages': [],
                    'document_type': self._detect_document_type(pdf),
                    'tables': [],
                    'forms': [],
                    'semantic_structure': {}
                }

                # Process pages in parallel
                with ProcessPoolExecutor() as executor:
                    futures = []
                    for page_num in range(len(pdf.pages)):
                        future = executor.submit(
                            self._process_page,
                            pdf_path,
                            page_num,
                            output_dir
                        )
                        futures.append(future)

                    # Collect results
                    for future in futures:
                        page_content = future.result()
                        if page_content:
                            document_structure['pages'].append(page_content)

                # Post-process and structure the content
                document_structure['semantic_structure'] = self._build_semantic_structure(
                    document_structure['pages']
                )

                # Clean and validate the extracted data
                document_structure = self._clean_and_validate(document_structure)

                # Save structured output
                output_path = Path(output_dir) / 'structured_content.json'
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(document_structure, f, indent=2)

                return document_structure

        except Exception as e:
            self.logger.error(f"Error parsing document: {str(e)}", exc_info=True)
            raise

    def _process_page(self, pdf_path: str, page_num: int, output_dir: str) -> Dict[str, Any]:
        """Process a single page of the PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num]

                # Extract and categorize all elements on the page
                elements = []

                # Extract text with positioning
                text_elements = self._extract_text_elements(page)
                elements.extend(text_elements)

                # Extract tables
                tables = self._extract_tables(page)
                elements.extend(tables)

                # Extract images
                images = self._extract_images(pdf_path, page_num, output_dir)
                elements.extend(images)

                # Analyze spatial relationships
                layout = self._analyze_layout(elements)

                # Identify semantic regions
                semantic_regions = self._identify_semantic_regions(elements)

                return {
                    'page_number': page_num + 1,
                    'elements': elements,
                    'layout': layout,
                    'semantic_regions': semantic_regions
                }

        except Exception as e:
            self.logger.error(f"Error processing page {page_num + 1}: {str(e)}", exc_info=True)
            return None

    def _extract_text_elements(self, page) -> List[PageElement]:
        """Extract and categorize text elements with their spatial information"""
        elements = []
        words = page.extract_words(keep_blank_chars=True, y_tolerance=3)

        for word in words:
            # Perform NER on the text
            doc = self.nlp(word['text'])

            # Determine element type and confidence
            element_type = 'text'
            confidence = 1.0

            # Check for special patterns
            for pattern_name, pattern in self.patterns.items():
                if re.match(pattern, word['text']):
                    element_type = pattern_name
                    break

            # Create element with positioning
            element = PageElement(
                content=word['text'],
                element_type=element_type,
                confidence=confidence,
                bounding_box=BoundingBox(
                    x0=word['x0'],
                    y0=word['top'],
                    x1=word['x1'],
                    y1=word['bottom']
                ),
                metadata={
                    'font': word.get('font', ''),
                    'size': word.get('size', 0),
                    'entities': [(ent.text, ent.label_) for ent in doc.ents]
                }
            )
            elements.append(element)

        return elements

    def _extract_tables(self, page) -> List[PageElement]:
        """Extract tables with enhanced structure detection"""
        tables = []
        try:
            for table in page.find_tables():
                df = pd.DataFrame(table.extract())

                # Clean and standardize table data
                df = self._clean_table_data(df)

                element = PageElement(
                    content=df.to_dict('records'),
                    element_type='table',
                    confidence=0.9,
                    bounding_box=BoundingBox(
                        x0=table.bbox[0],
                        y0=table.bbox[1],
                        x1=table.bbox[2],
                        y1=table.bbox[3]
                    ),
                    metadata={
                        'columns': df.columns.tolist(),
                        'num_rows': len(df),
                        'table_type': self._detect_table_type(df)
                    }
                )
                tables.append(element)
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}", exc_info=True)

        return tables

    @staticmethod
    def _clean_table_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize table data"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Standardize column names
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]

        # Clean cell values
        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        return df

    def _extract_metadata(self, pdf) -> Dict[str, Any]:
        """Extract metadata from PDF document"""
        try:
            metadata = {
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'creation_date': pdf.metadata.get('CreationDate', ''),
                'num_pages': len(pdf.pages),
                'file_size': Path(pdf.stream.name).stat().st_size if hasattr(pdf.stream, 'name') else None
            }
            return metadata
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}", exc_info=True)
            return {}

    def _detect_document_type(self, pdf) -> str:
        """Detect document type based on content and structure"""
        try:
            # Sample first few pages
            sample_pages = min(3, len(pdf.pages))
            text_samples = []
            for i in range(sample_pages):
                text = pdf.pages[i].extract_text()
                text_samples.append(text.lower())

            # Define document type patterns
            patterns = {
                'academic_paper': r'\b(?:abstract|introduction|methodology|conclusion)\b',
                'report': r'\b(?:executive\s+summary|findings|recommendations)\b',
                'manual': r'\b(?:instructions?|guide|manual|step\s+\d+)\b',
                'legal_document': r'\b(?:pursuant|hereby|agreement|contract|terms)\b',
                'financial_document': r'\b(?:balance\s+sheet|income\s+statement|cash\s+flow)\b'
            }

            # Score each document type
            scores = {doc_type: 0 for doc_type in patterns}

            for text in text_samples:
                for doc_type, pattern in patterns.items():
                    if re.search(pattern, text):
                        scores[doc_type] += 1

            # Return highest scoring type or 'general' if no clear match
            max_score = max(scores.values())
            if max_score > 0:
                return max(scores.items(), key=lambda x: x[1])[0]
            return 'general'

        except Exception as e:
            self.logger.error(f"Error detecting document type: {str(e)}", exc_info=True)
            return 'unknown'

    def _build_semantic_structure(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build semantic structure from extracted elements"""
        structure = {
            'sections': [],
            'hierarchical_structure': {},
            'key_entities': {},
            'relationships': []
        }

        current_section = None

        for page in pages:
            for element in page['elements']:
                # Identify section headers
                if element.element_type == 'section_headers':
                    current_section = {
                        'title': element.content,
                        'content': [],
                        'subsections': []
                    }
                    structure['sections'].append(current_section)

                # Build relationships between elements
                elif current_section is not None:
                    current_section['content'].append(element)

                # Extract and categorize entities
                if element.metadata.get('entities'):
                    for entity_text, entity_type in element.metadata['entities']:
                        if entity_type not in structure['key_entities']:
                            structure['key_entities'][entity_type] = []
                        structure['key_entities'][entity_type].append(entity_text)

        return structure

    def _clean_and_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted data"""
        try:
            # Clean table data
            if 'tables' in data:
                data['tables'] = [
                    self._clean_table_data(pd.DataFrame(table['content'])) for table in data['tables']
                ]

            # Standardize dates
            data = self._standardize_dates(data)

            # Remove duplicates
            data = self._remove_duplicates(data)

            # Validate data types
            data = self._validate_data_types(data)

            return data

        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}", exc_info=True)
            return data

    def _standardize_dates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize date formats in the data"""
        try:
            def convert_date(text):
                if not isinstance(text, str):
                    return text

                # Add date format patterns and conversion logic here
                date_patterns = [
                    (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),
                    (r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\3-\1-\2'),
                ]

                for pattern, format_str in date_patterns:
                    if re.match(pattern, text):
                        return re.sub(pattern, format_str, text)
                return text

            def process_dict(d):
                for key, value in d.items():
                    if isinstance(value, dict):
                        d[key] = process_dict(value)
                    elif isinstance(value, list):
                        d[key] = [process_dict(item) if isinstance(item, dict)
                                  else convert_date(item) for item in value]
                    else:
                        d[key] = convert_date(value)
                return d

            return process_dict(data)

        except Exception as e:
            self.logger.error(f"Error standardizing dates: {str(e)}", exc_info=True)
            return data

    def _remove_duplicates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate content from extracted data"""
        try:
            def hash_content(content):
                if isinstance(content, (str, int, float)):
                    return hashlib.md5(str(content).encode()).hexdigest()
                return None

            def deduplicate_list(items):
                seen_hashes = set()
                unique_items = []

                for item in items:
                    if isinstance(item, dict):
                        item_hash = hash_content(str(sorted(item.items())))
                    else:
                        item_hash = hash_content(item)

                    if item_hash and item_hash not in seen_hashes:
                        seen_hashes.add(item_hash)
                        unique_items.append(item)

                return unique_items

            def process_dict(d):
                for key, value in d.items():
                    if isinstance(value, dict):
                        d[key] = process_dict(value)
                    elif isinstance(value, list):
                        d[key] = deduplicate_list(value)
                return d

            return process_dict(data)

        except Exception as e:
            self.logger.error(f"Error removing duplicates: {str(e)}", exc_info=True)
            return data

    def _validate_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert data types"""
        try:
            def convert_type(value):
                if isinstance(value, str):
                    # Try converting to number
                    try:
                        if '.' in value:
                            return float(value)
                        return int(value)
                    except ValueError:
                        pass

                    # Try converting to boolean
                    if value.lower() in ('true', 'false'):
                        return value.lower() == 'true'

                return value

            def process_dict(d):
                for key, value in d.items():
                    if isinstance(value, dict):
                        d[key] = process_dict(value)
                    elif isinstance(value, list):
                        d[key] = [process_dict(item) if isinstance(item, dict)
                                  else convert_type(item) for item in value]
                    else:
                        d[key] = convert_type(value)
                return d

            return process_dict(data)

        except Exception as e:
            self.logger.error(f"Error validating data types: {str(e)}", exc_info=True)
            return data

    def _extract_images(self, pdf_path: str, page_num: int, output_dir: str) -> List[PageElement]:
        """Extract images from PDF page"""
        try:
            images = []
            doc = fitz.open(pdf_path)
            page = doc[page_num]

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                if base_image:
                    # Save the image to output_dir
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"image_page{page_num + 1}_{img_index}.{image_ext}"
                    image_path = Path(output_dir) / image_filename
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_bytes)

                    # Get image bounding box
                    bbox = self._get_image_bbox(page, xref)

                    # Create image element
                    element = PageElement(
                        content=str(image_path),
                        element_type='image',
                        confidence=0.9,
                        bounding_box=bbox,
                        metadata={
                            'width': base_image.get('width'),
                            'height': base_image.get('height'),
                            'colorspace': base_image.get('colorspace')
                        }
                    )
                    images.append(element)

            return images

        except Exception as e:
            self.logger.error(f"Error extracting images from page {page_num}: {str(e)}", exc_info=True)
            return []

    def _get_image_bbox(self, page, xref) -> BoundingBox:
        """Helper method to get the bounding box of an image given its xref"""
        try:
            for img in page.get_images(full=True):
                if img[0] == xref:
                    # The position is at index 7 in the tuple
                    bbox = img[7]
                    return BoundingBox(
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3]
                    )
        except Exception as e:
            self.logger.error(f"Error getting image bounding box: {str(e)}", exc_info=True)
        # Return a default bounding box if not found
        return BoundingBox(0, 0, 0, 0)

    def _detect_table_type(self, df: pd.DataFrame) -> str:
        """Detect the type of table based on its content and structure."""
        try:
            # Get column names and convert to lowercase for analysis
            columns = [str(col).lower() for col in df.columns]

            # Get sample of data for pattern analysis
            sample_data = df.head(10).values.flatten()
            sample_data = [str(val).lower() for val in sample_data if pd.notna(val)]

            # Define pattern detectors
            financial_patterns = {
                'columns': ['amount', 'total', 'balance', 'revenue', 'cost', 'price', 'fee', 'tax', 'credit', 'debit'],
                'data': r'^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$'
            }

            property_patterns = {
                'columns': ['address', 'location', 'property', 'size', 'area', 'price', 'value'],
                'data': r'\b(?:sq ft|acre|property|house|building)\b'
            }

            contact_patterns = {
                'columns': ['name', 'email', 'phone', 'address', 'contact'],
                'data': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            }

            # Check for table characteristics
            characteristics = {
                'num_columns': len(df.columns),
                'num_rows': len(df),
                'numeric_ratio': df.select_dtypes(include=['number']).size / df.size if df.size > 0 else 0,
                'empty_ratio': df.isna().sum().sum() / df.size if df.size > 0 else 0
            }

            # Score each table type based on patterns
            scores = {
                'financial': 0,
                'property_listing': 0,
                'contact_list': 0,
                'data_matrix': 0,
                'summary_table': 0
            }

            # Check column patterns
            for col in columns:
                if any(pattern in col for pattern in financial_patterns['columns']):
                    scores['financial'] += 2
                if any(pattern in col for pattern in property_patterns['columns']):
                    scores['property_listing'] += 2
                if any(pattern in col for pattern in contact_patterns['columns']):
                    scores['contact_list'] += 2

            # Check data patterns
            financial_data_pattern = re.compile(financial_patterns['data'])
            property_data_pattern = re.compile(property_patterns['data'])
            contact_data_pattern = re.compile(contact_patterns['data'])

            for value in sample_data:
                if financial_data_pattern.match(str(value)):
                    scores['financial'] += 1
                if property_data_pattern.search(str(value)):
                    scores['property_listing'] += 1
                if contact_data_pattern.search(str(value)):
                    scores['contact_list'] += 1

            # Check structural characteristics
            if characteristics['numeric_ratio'] > 0.7:
                scores['data_matrix'] += 3
            if characteristics['num_columns'] <= 3 and characteristics['num_rows'] <= 10:
                scores['summary_table'] += 2

            # Get the table type with highest score
            table_type = max(scores.items(), key=lambda x: x[1])[0]

            # If no clear pattern is detected, classify based on structure
            if max(scores.values()) == 0:
                if characteristics['numeric_ratio'] > 0.5:
                    return 'numerical_data'
                if characteristics['num_columns'] == 2:
                    return 'key_value_pairs'
                return 'general_table'

            return table_type

        except Exception as e:
            self.logger.error(f"Error detecting table type: {str(e)}", exc_info=True)
            return 'unknown'

    def _analyze_layout(self, elements: List[PageElement]) -> Dict[str, Any]:
        """
        Analyze spatial relationships between elements on the page.

        Determines element positioning, alignment, and hierarchical relationships
        based on spatial analysis.

        Args:
            elements: List of PageElement objects with positioning information

        Returns:
            Dict containing layout analysis results
        """
        try:
            layout_analysis = {
                'columns': [],
                'rows': [],
                'hierarchical_groups': [],
                'alignment_groups': {
                    'left_aligned': [],
                    'right_aligned': [],
                    'center_aligned': [],
                },
                'spatial_relationships': []
            }

            if not elements:
                return layout_analysis

            # Sort elements by position
            elements_by_x = sorted(elements, key=lambda e: e.bounding_box.x0)
            elements_by_y = sorted(elements, key=lambda e: e.bounding_box.y0)

            # Detect columns (vertical alignment)
            x_positions = [e.bounding_box.x0 for e in elements]
            x_clusters = self._cluster_positions(x_positions, threshold=20)

            for cluster in x_clusters:
                column_elements = [
                    e for e in elements
                    if any(abs(e.bounding_box.x0 - x) < 20 for x in cluster)
                ]
                if column_elements:
                    layout_analysis['columns'].append({
                        'x_position': sum(cluster) / len(cluster),
                        'elements': column_elements
                    })

            # Detect rows (horizontal alignment)
            y_positions = [e.bounding_box.y0 for e in elements]
            y_clusters = self._cluster_positions(y_positions, threshold=10)

            for cluster in y_clusters:
                row_elements = [
                    e for e in elements
                    if any(abs(e.bounding_box.y0 - y) < 10 for y in cluster)
                ]
                if row_elements:
                    layout_analysis['rows'].append({
                        'y_position': sum(cluster) / len(cluster),
                        'elements': row_elements
                    })

            # Analyze alignment
            page_width = max(e.bounding_box.x1 for e in elements)

            for element in elements:
                # Left alignment
                if abs(element.bounding_box.x0 - min(x_positions)) < 20:
                    layout_analysis['alignment_groups']['left_aligned'].append(element)

                # Right alignment
                if abs(element.bounding_box.x1 - page_width) < 20:
                    layout_analysis['alignment_groups']['right_aligned'].append(element)

                # Center alignment
                element_center = (element.bounding_box.x0 + element.bounding_box.x1) / 2
                if abs(element_center - page_width / 2) < 30:
                    layout_analysis['alignment_groups']['center_aligned'].append(element)

            # Detect hierarchical relationships
            for i, element in enumerate(elements):
                for other in elements[i + 1:]:
                    # Check for containment
                    if (element.bounding_box.x0 <= other.bounding_box.x0 and
                        element.bounding_box.x1 >= other.bounding_box.x1 and
                        element.bounding_box.y0 <= other.bounding_box.y0 and
                        element.bounding_box.y1 >= other.bounding_box.y1):
                        layout_analysis['hierarchical_groups'].append({
                            'parent': element,
                            'child': other
                        })

                    # Check for spatial relationships
                    spatial_rel = self._determine_spatial_relationship(element, other)
                    if spatial_rel:
                        layout_analysis['spatial_relationships'].append(spatial_rel)

            return layout_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing layout: {str(e)}", exc_info=True)
            return {}

    def _identify_semantic_regions(self, elements: List[PageElement]) -> List[Dict[str, Any]]:
        """
        Identify semantic regions in the document based on content and layout.

        Groups elements into meaningful semantic regions like headers, paragraphs,
        lists, etc.

        Args:
            elements: List of PageElement objects

        Returns:
            List of dictionaries describing semantic regions
        """
        try:
            semantic_regions = []
            current_region = None

            # Sort elements by vertical position
            elements = sorted(elements, key=lambda e: (e.bounding_box.y0, e.bounding_box.x0))

            # Define region type detection rules
            header_patterns = [
                r'^[A-Z\s]+$',  # All caps
                r'^[A-Z][A-Za-z\s]+:$',  # Title case with colon
                r'^[\d\.]+\s+[A-Z]'  # Numbered headers
            ]

            list_patterns = [
                r'^\s*[\-\•]\s',  # Bullet points
                r'^\s*\d+\.',  # Numbered lists
                r'^\s*[a-z]\.',  # Alphabetical lists
            ]

            for element in elements:
                # Skip elements that are part of tables
                if element.element_type == 'table':
                    if current_region:
                        semantic_regions.append(current_region)
                        current_region = None
                    semantic_regions.append({
                        'type': 'table',
                        'elements': [element],
                        'bbox': element.bounding_box,
                        'metadata': element.metadata
                    })
                    continue

                # Detect headers
                is_header = False
                if element.element_type == 'text':
                    text = element.content.strip()
                    if any(re.match(pattern, text) for pattern in header_patterns):
                        is_header = True

                    if is_header:
                        if current_region:
                            semantic_regions.append(current_region)
                        current_region = {
                            'type': 'header',
                            'elements': [element],
                            'bbox': element.bounding_box,
                            'level': self._detect_header_level(element)
                        }
                        continue

                # Detect lists
                is_list_item = False
                if element.element_type == 'text':
                    text = element.content.strip()
                    if any(re.match(pattern, text) for pattern in list_patterns):
                        is_list_item = True

                    if is_list_item:
                        if current_region and current_region['type'] != 'list':
                            semantic_regions.append(current_region)
                            current_region = None

                        if not current_region:
                            current_region = {
                                'type': 'list',
                                'elements': [],
                                'bbox': element.bounding_box,
                                'list_type': 'bullet' if '•' in text else 'numbered'
                            }
                        current_region['elements'].append(element)
                        continue

                # Group paragraph text
                if element.element_type == 'text' and not is_header and not is_list_item:
                    if not current_region or current_region['type'] != 'paragraph':
                        if current_region:
                            semantic_regions.append(current_region)
                        current_region = {
                            'type': 'paragraph',
                            'elements': [],
                            'bbox': element.bounding_box
                        }
                    current_region['elements'].append(element)
                    # Update bounding box
                    current_region['bbox'] = self._merge_bounding_boxes(
                        current_region['bbox'],
                        element.bounding_box
                    )

            # Add final region
            if current_region:
                semantic_regions.append(current_region)

            # Post-process regions
            self._post_process_regions(semantic_regions)

            return semantic_regions

        except Exception as e:
            self.logger.error(f"Error identifying semantic regions: {str(e)}", exc_info=True)
            return []

    def _cluster_positions(self, positions: List[float], threshold: float) -> List[List[float]]:
        """Helper method to cluster similar positions together"""
        if not positions:
            return []

        clusters = [[positions[0]]]

        for pos in positions[1:]:
            merged = False
            for cluster in clusters:
                if abs(sum(cluster) / len(cluster) - pos) < threshold:
                    cluster.append(pos)
                    merged = True
                    break
            if not merged:
                clusters.append([pos])

        return clusters

    def _determine_spatial_relationship(
        self,
        element1: PageElement,
        element2: PageElement
    ) -> Optional[Dict[str, Any]]:
        """Helper method to determine spatial relationship between elements"""
        if not (element1 and element2):
            return None

        dx = element2.bounding_box.x0 - element1.bounding_box.x1
        dy = element2.bounding_box.y0 - element1.bounding_box.y1

        relationship = {
            'element1': element1,
            'element2': element2,
            'distance_x': dx,
            'distance_y': dy
        }

        if abs(dx) < 20 and dy > 0:
            relationship['type'] = 'vertical_flow'
        elif abs(dy) < 10 and dx > 0:
            relationship['type'] = 'horizontal_flow'
        else:
            relationship['type'] = 'disconnected'

        return relationship

    def _detect_header_level(self, element: PageElement) -> int:
        """Helper method to detect header level based on formatting"""
        if not element.metadata.get('size'):
            return 1

        font_size = element.metadata['size']

        # Simple heuristic based on font size
        if font_size >= 20:
            return 1
        elif font_size >= 16:
            return 2
        elif font_size >= 14:
            return 3
        else:
            return 4

    def _merge_bounding_boxes(self, box1: BoundingBox, box2: BoundingBox) -> BoundingBox:
        """Helper method to merge two bounding boxes"""
        return BoundingBox(
            x0=min(box1.x0, box2.x0),
            y0=min(box1.y0, box2.y0),
            x1=max(box1.x1, box2.x1),
            y1=max(box1.y1, box2.y1)
        )

    def _post_process_regions(self, regions: List[Dict[str, Any]]) -> None:
        """
        Helper method to post-process and refine detected regions.
        Handles merging of related regions and refinement of region types.
        """
        # Merge adjacent paragraphs with similar formatting
        i = 0
        while i < len(regions) - 1:
            current = regions[i]
            next_region = regions[i + 1]

            if (current['type'] == 'paragraph' and
                next_region['type'] == 'paragraph' and
                self._should_merge_paragraphs(current, next_region)):

                # Merge elements and update bounding box
                current['elements'].extend(next_region['elements'])
                current['bbox'] = self._merge_bounding_boxes(
                    current['bbox'],
                    next_region['bbox']
                )

                # Remove the merged region
                regions.pop(i + 1)
            else:
                i += 1

        # Identify section relationships
        section_stack = []
        for region in regions:
            if region['type'] == 'header':
                # Pop sections with higher or equal level from stack
                while (section_stack and
                       section_stack[-1]['level'] >= region['level']):
                    section_stack.pop()

                # Add parent relationship if stack isn't empty
                if section_stack:
                    region['parent_section'] = section_stack[-1]
                    if 'subsections' not in section_stack[-1]:
                        section_stack[-1]['subsections'] = []
                    section_stack[-1]['subsections'].append(region)

                section_stack.append(region)

        # Detect and label special regions
        for region in regions:
            if region['type'] == 'paragraph':
                # Check for special paragraph types
                first_element = region['elements'][0]
                text = first_element.content.strip()

                # Detect notes/footnotes
                if re.match(r'^\d+\.\s+', text) or text.startswith('*') or text.startswith('Note:'):
                    region['subtype'] = 'note'

                # Detect citations
                elif re.match(r'^\[\d+\]', text) or re.match(r'^\(\d{4}\)', text):
                    region['subtype'] = 'citation'

                # Detect quotes
                elif text.startswith('"') and text.endswith('"'):
                    region['subtype'] = 'quote'

        # Calculate statistics and metadata for regions
        for region in regions:
            region['statistics'] = self._calculate_region_statistics(region)
            region['metadata'] = self._extract_region_metadata(region)

    def _should_merge_paragraphs(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> bool:
        """
        Determine if two paragraph regions should be merged based on their properties.

        Args:
            region1: First paragraph region
            region2: Second paragraph region

        Returns:
            bool: True if regions should be merged
        """
        try:
            # Check vertical distance
            vertical_gap = region2['bbox'].y0 - region1['bbox'].y1
            if vertical_gap > 20:  # Too far apart
                return False

            # Check formatting consistency
            if not self._has_similar_formatting(region1['elements'], region2['elements']):
                return False

            # Check alignment
            if abs(region1['bbox'].x0 - region2['bbox'].x0) > 10:  # Different indentation
                return False

            # Check if they're part of the same semantic flow
            last_element = region1['elements'][-1]
            first_element = region2['elements'][0]

            # Don't merge if first region ends with sentence-ending punctuation
            if last_element.content.strip().endswith(('.', '!', '?')):
                # Check for exceptions like "e.g.", "i.e.", etc.
                if not re.search(r'\b[A-Za-z]\.[A-Za-z]\.$', last_element.content.strip()):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking paragraph merge: {str(e)}", exc_info=True)
            return False

    def _has_similar_formatting(self, elements1: List[PageElement], elements2: List[PageElement]) -> bool:
        """
        Check if two sets of elements have similar formatting.

        Args:
            elements1: First set of elements
            elements2: Second set of elements

        Returns:
            bool: True if formatting is similar
        """
        try:
            # Get formatting attributes from first elements
            def get_formatting(element):
                return {
                    'font': element.metadata.get('font'),
                    'size': element.metadata.get('size'),
                    'style': element.metadata.get('style', '')
                }

            format1 = get_formatting(elements1[0])
            format2 = get_formatting(elements2[0])

            # Compare font attributes
            if format1['font'] != format2['font']:
                return False

            # Allow small variations in size
            if abs(format1['size'] - format2['size']) > 1:
                return False

            # Compare styles
            if format1['style'] != format2['style']:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error comparing formatting: {str(e)}", exc_info=True)
            return False

    def _calculate_region_statistics(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistical information about a region.

        Args:
            region: Region dictionary containing elements and metadata

        Returns:
            Dictionary containing statistical information
        """
        stats = {
            'num_elements': len(region['elements']),
            'total_area': 0,
            'avg_font_size': 0,
            'text_density': 0,
            'word_count': 0
        }

        try:
            # Calculate area
            width = region['bbox'].x1 - region['bbox'].x0
            height = region['bbox'].y1 - region['bbox'].y0
            stats['total_area'] = width * height

            # Calculate averages
            font_sizes = []
            total_text_length = 0

            for element in region['elements']:
                if element.metadata.get('size'):
                    font_sizes.append(element.metadata['size'])

                if element.content:
                    total_text_length += len(element.content)
                    stats['word_count'] += len(element.content.split())

            if font_sizes:
                stats['avg_font_size'] = sum(font_sizes) / len(font_sizes)

            if stats['total_area'] > 0:
                stats['text_density'] = total_text_length / stats['total_area']

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating region statistics: {str(e)}", exc_info=True)
            return stats

    def _extract_region_metadata(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and organize metadata for a region.

        Args:
            region: Region dictionary containing elements and metadata

        Returns:
            Dictionary containing region metadata
        """
        metadata = {
            'language': None,
            'reading_level': None,
            'entities': [],
            'keywords': [],
            'semantic_role': None
        }

        try:
            # Combine all text in region
            text = ' '.join(e.content for e in region['elements'] if e.content)

            # Detect language
            metadata['language'] = self.nlp.meta['lang']

            # Extract entities
            doc = self.nlp(text)
            metadata['entities'] = [
                {'text': ent.text, 'label': ent.label_}
                for ent in doc.ents
            ]

            # Extract keywords (using basic frequency for now)
            words = [token.text.lower() for token in doc
                     if not token.is_stop and not token.is_punct]
            word_freq = Counter(words)
            metadata['keywords'] = [
                word for word, count in word_freq.most_common(5)
            ]

            # Determine semantic role
            metadata['semantic_role'] = self._determine_semantic_role(region)

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting region metadata: {str(e)}", exc_info=True)
            return metadata

    def _determine_semantic_role(self, region: Dict[str, Any]) -> str:
        """
        Determine the semantic role of a region based on its content and context.

        Args:
            region: Region dictionary containing elements and metadata

        Returns:
            str: Semantic role of the region
        """
        try:
            if region['type'] == 'header':
                return 'section_title'

            if region.get('subtype') in ['note', 'citation', 'quote']:
                return region['subtype']

            # Get text content
            text = ' '.join(e.content for e in region['elements'] if e.content)

            # Check for common patterns
            if re.search(r'\b(?:conclude|summary|conclusion)\b', text.lower()):
                return 'conclusion'

            if re.search(r'\b(?:introduce|introduction)\b', text.lower()):
                return 'introduction'

            if re.search(r'\b(?:method|methodology)\b', text.lower()):
                return 'methodology'

            if re.search(r'\b(?:result|finding)\b', text.lower()):
                return 'results'

            if re.search(r'\b(?:discuss|implication)\b', text.lower()):
                return 'discussion'

            # Default to body text if no specific role is identified
            return 'body_text'

        except Exception as e:
            self.logger.error(f"Error determining semantic role: {str(e)}", exc_info=True)
            return 'unknown'

if __name__ == "__main__":
    parser = PDFSemanticParser()
    result = parser.parse_document("input.pdf", "output_dir")