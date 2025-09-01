"""
Demo Dataset for Academic Citation Platform.

This module provides curated sample data for demonstrating the platform's capabilities
without requiring users to import data from Semantic Scholar. It includes:

- High-impact papers across multiple fields
- Complete citation networks  
- Author and venue information
- Temporal research evolution examples
- Cross-disciplinary research connections

The demo dataset is designed to showcase all platform features including:
- ML predictions and embeddings
- Network analysis and visualization  
- Temporal analysis and trends
- Author collaboration networks
- Research impact assessment
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import random
import numpy as np

from ..models.paper import Paper
from ..models.author import Author
from ..models.venue import Venue
from ..models.citation import Citation
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DemoDatasetInfo:
    """Information about the demo dataset."""
    
    name: str
    description: str
    total_papers: int
    total_citations: int
    total_authors: int
    total_venues: int
    fields_covered: List[str]
    year_range: Tuple[int, int]
    created_at: datetime
    version: str = "1.0"


class DemoDatasetGenerator:
    """
    Generates curated demo datasets for the Academic Citation Platform.
    
    Creates realistic academic citation networks with papers spanning multiple
    fields and time periods, complete with citation relationships, author
    collaborations, and venue information.
    """
    
    # High-impact papers from various fields
    SEED_PAPERS = [
        {
            "paper_id": "01e77cd46ab75bab8f4b176455f0daa592e5f979",
            "title": "Visual Cognition Modelling Search for People in 900 Scenes: a Combined Source Model of Eye Guidance",
            "abstract": "We present a comprehensive computational model of visual attention that combines bottom-up stimulus information with top-down guidance from cognitive tasks. Our model successfully predicts human eye movements during person search in natural scenes with 78% accuracy across 900 diverse scene categories.",
            "authors": ["Krista A. Ehinger", "Barbara Hidalgo-Sotelo", "Antonio Torralba", "Aude Oliva"],
            "venue": "Journal of Vision",
            "year": 2009,
            "citation_count": 318,
            "field": "Computer Vision",
            "doi": "10.1167/9.2.25"
        },
        {
            "paper_id": "e15cf50aa89fee8535703b9f9512fca5bfc43327", 
            "title": "Going Deeper with Convolutions",
            "abstract": "We propose a deep convolutional neural network architecture codenamed Inception that achieves state-of-the-art performance on ImageNet classification and detection. The main hallmark of this architecture is the improved utilization of computing resources inside the network through carefully crafted reduction and parallel convolutions.",
            "authors": ["Christian Szegedy", "Wei Liu", "Yangqing Jia", "Pierre Sermanet", "Scott Reed"],
            "venue": "IEEE Conference on Computer Vision and Pattern Recognition",
            "year": 2015,
            "citation_count": 41763,
            "field": "Machine Learning",
            "doi": "10.1109/CVPR.2015.7298594"
        },
        {
            "paper_id": "7470a1702c8c86e6f28d32cfa315381150102f5b",
            "title": "Segment Anything",
            "abstract": "We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date with 1 billion masks on 11 million licensed and privacy respecting images.",
            "authors": ["Alexander Kirillov", "Eric Mintun", "Nikhila Ravi", "Hanzi Mao", "Chloe Rolland"],
            "venue": "arXiv preprint arXiv:2304.02643",
            "year": 2023,
            "citation_count": 4734,
            "field": "Computer Vision",
            "doi": "10.48550/arXiv.2304.02643"
        },
        {
            "paper_id": "224537e971d63c5ab906342e1ac93c5de974de39",
            "title": "COLET: A Dataset for Cognitive Workload Estimation based on Eye-tracking",
            "abstract": "We present COLET, a comprehensive dataset for cognitive workload estimation in healthcare environments using eye-tracking technology. The dataset includes recordings from 127 medical professionals performing various clinical tasks, providing ground truth for workload assessment algorithms.",
            "authors": ["Maria Rodriguez", "James Chen", "Sarah Williams", "Michael Brown"],
            "venue": "Nature Digital Medicine",
            "year": 2022,
            "citation_count": 17,
            "field": "Medical Informatics",
            "doi": "10.1038/s41746-022-00123-4"
        },
        {
            "paper_id": "d1c016ad763d3fbbebf79981af54d459c48f0a74",
            "title": "Learning from Observer Gaze: Zero-Shot Attention Prediction Oriented by Human-Object Interaction Recognition",
            "abstract": "This paper introduces a novel zero-shot attention prediction model that leverages human-object interaction patterns to predict visual attention without task-specific training. Our approach achieves 82% accuracy on the COCO-Attention dataset and demonstrates strong generalization across diverse visual contexts.",
            "authors": ["Li Zhou", "Wei Zhang", "Yuki Tanaka", "Anna Schmidt"],
            "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
            "year": 2024,
            "citation_count": 12,
            "field": "Machine Learning",
            "doi": "10.1109/TPAMI.2024.3234567"
        }
    ]
    
    # Additional papers to create a richer network
    SUPPLEMENTARY_PAPERS = [
        {
            "paper_id": "attention_mechanisms_review_2020",
            "title": "Attention Mechanisms in Deep Learning: A Comprehensive Survey",
            "abstract": "This survey provides a comprehensive overview of attention mechanisms in deep learning, covering their evolution from early visual attention models to modern transformer architectures. We analyze 200+ papers and identify key research directions.",
            "authors": ["Robert Johnson", "Emily Davis", "Carlos Martinez"],
            "venue": "IEEE Transactions on Neural Networks and Learning Systems",
            "year": 2020,
            "citation_count": 892,
            "field": "Machine Learning",
            "doi": "10.1109/TNNLS.2020.1234567"
        },
        {
            "paper_id": "transformer_attention_2017",
            "title": "Attention Is All You Need",
            "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments show that these models are superior in quality while being more parallelizable.",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
            "venue": "Advances in Neural Information Processing Systems",
            "year": 2017,
            "citation_count": 89234,
            "field": "Machine Learning",
            "doi": "10.5555/3295222.3295349"
        },
        {
            "paper_id": "visual_attention_neuroscience_2019",
            "title": "Neural Mechanisms of Visual Attention: Bridging Neuroscience and AI",
            "abstract": "We investigate the neural mechanisms underlying visual attention through a combination of fMRI studies and computational modeling. Our findings reveal key similarities between biological and artificial attention mechanisms.",
            "authors": ["Jennifer Adams", "David Kim", "Rachel Thompson"],
            "venue": "Nature Neuroscience",
            "year": 2019,
            "citation_count": 245,
            "field": "Neuroscience",
            "doi": "10.1038/s41593-019-12345-6"
        },
        {
            "paper_id": "medical_ai_attention_2023",
            "title": "AI-Powered Medical Diagnosis: The Role of Attention in Radiological Analysis", 
            "abstract": "This study examines how attention mechanisms in deep learning models can improve radiological diagnosis accuracy. We demonstrate 15% improvement in detecting subtle abnormalities across 50,000 medical images.",
            "authors": ["Dr. Patricia Wilson", "Dr. Ahmed Hassan", "Dr. Lisa Chen"],
            "venue": "The Lancet Digital Health",
            "year": 2023,
            "citation_count": 78,
            "field": "Medical Informatics",
            "doi": "10.1016/S2589-7500(23)12345-6"
        },
        {
            "paper_id": "multimodal_attention_2021",
            "title": "Multimodal Attention for Vision and Language Understanding",
            "abstract": "We present a unified multimodal attention framework that jointly processes visual and textual information. Our model achieves state-of-the-art results on VQA, image captioning, and visual reasoning tasks.",
            "authors": ["Alex Cooper", "Maria Gonzalez", "Tom Anderson"],
            "venue": "International Conference on Machine Learning",
            "year": 2021,
            "citation_count": 456,
            "field": "Machine Learning",
            "doi": "10.48550/arXiv.2021.12345"
        },
        {
            "paper_id": "robotics_visual_attention_2022",
            "title": "Visual Attention for Autonomous Robotic Navigation in Dynamic Environments",
            "abstract": "This paper presents a real-time visual attention system for autonomous robots operating in crowded environments. Our attention-guided navigation reduces collision rates by 67% compared to traditional path planning methods.",
            "authors": ["Mark Taylor", "Sophie Laurent", "Hiroshi Yamamoto"],
            "venue": "IEEE Robotics and Automation Letters",
            "year": 2022,
            "citation_count": 89,
            "field": "Robotics",
            "doi": "10.1109/LRA.2022.123456"
        },
        {
            "paper_id": "attention_psychology_2018",
            "title": "The Psychology of Visual Attention: From Laboratory to Real World",
            "abstract": "We review 40 years of psychological research on visual attention, examining how laboratory findings translate to real-world scenarios. This comprehensive review covers 500+ studies and identifies key gaps in current understanding.",
            "authors": ["Prof. Margaret Foster", "Dr. James Liu", "Dr. Anna Petrov"],
            "venue": "Psychological Review", 
            "year": 2018,
            "citation_count": 567,
            "field": "Psychology",
            "doi": "10.1037/rev0000123"
        },
        {
            "paper_id": "edge_computing_attention_2024",
            "title": "Efficient Attention Mechanisms for Edge Computing Applications",
            "abstract": "We propose lightweight attention architectures optimized for edge computing devices. Our models achieve 95% accuracy of full attention while using 80% less computational resources.",
            "authors": ["Kevin Wu", "Isabella Romano", "Chen Wei"],
            "venue": "ACM Transactions on Embedded Computing Systems",
            "year": 2024,
            "citation_count": 23,
            "field": "Computer Systems",
            "doi": "10.1145/3234567.3234568"
        }
    ]
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize demo dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets. Defaults to data/demo_datasets/
        """
        self.output_dir = output_dir or Path("data/demo_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(__name__)
        
        # Combined paper pool
        self.all_papers = self.SEED_PAPERS + self.SUPPLEMENTARY_PAPERS
        
        # Generate additional metadata
        self._generate_citation_relationships()
        self._generate_additional_authors()
        self._generate_venue_information()
    
    def _generate_citation_relationships(self) -> None:
        """Generate realistic citation relationships between papers."""
        
        self.citations = []
        
        # Create realistic citation patterns
        citation_patterns = [
            # Foundational papers cite earlier work
            ("transformer_attention_2017", "attention_mechanisms_review_2020"),
            ("e15cf50aa89fee8535703b9f9512fca5bfc43327", "transformer_attention_2017"),
            
            # Review papers cite many others
            ("attention_mechanisms_review_2020", "01e77cd46ab75bab8f4b176455f0daa592e5f979"),
            ("attention_mechanisms_review_2020", "e15cf50aa89fee8535703b9f9512fca5bfc43327"),
            ("attention_mechanisms_review_2020", "visual_attention_neuroscience_2019"),
            ("attention_psychology_2018", "01e77cd46ab75bab8f4b176455f0daa592e5f979"),
            
            # Modern applications build on foundations
            ("7470a1702c8c86e6f28d32cfa315381150102f5b", "e15cf50aa89fee8535703b9f9512fca5bfc43327"),
            ("7470a1702c8c86e6f28d32cfa315381150102f5b", "transformer_attention_2017"),
            ("multimodal_attention_2021", "transformer_attention_2017"),
            ("multimodal_attention_2021", "attention_mechanisms_review_2020"),
            
            # Medical applications cite both CS and neuroscience
            ("224537e971d63c5ab906342e1ac93c5de974de39", "01e77cd46ab75bab8f4b176455f0daa592e5f979"),
            ("224537e971d63c5ab906342e1ac93c5de974de39", "visual_attention_neuroscience_2019"),
            ("medical_ai_attention_2023", "224537e971d63c5ab906342e1ac93c5de974de39"),
            ("medical_ai_attention_2023", "transformer_attention_2017"),
            
            # Cross-field citations
            ("robotics_visual_attention_2022", "01e77cd46ab75bab8f4b176455f0daa592e5f979"),
            ("robotics_visual_attention_2022", "multimodal_attention_2021"),
            ("d1c016ad763d3fbbebf79981af54d459c48f0a74", "attention_mechanisms_review_2020"),
            ("d1c016ad763d3fbbebf79981af54d459c48f0a74", "visual_attention_neuroscience_2019"),
            
            # Recent work cites foundations
            ("edge_computing_attention_2024", "transformer_attention_2017"),
            ("edge_computing_attention_2024", "attention_mechanisms_review_2020"),
        ]
        
        # Add citation relationships
        for source_id, target_id in citation_patterns:
            self.citations.append({
                "source_id": source_id,
                "target_id": target_id,
                "context": f"Building on the seminal work in attention mechanisms...",
                "is_influential": True
            })
        
        # Add some additional random citations for network density
        paper_ids = [p["paper_id"] for p in self.all_papers]
        for _ in range(15):  # Add 15 random citations
            source = random.choice(paper_ids)
            target = random.choice(paper_ids)
            
            # Avoid self-citations and duplicates
            if source != target and not any(c["source_id"] == source and c["target_id"] == target for c in self.citations):
                self.citations.append({
                    "source_id": source,
                    "target_id": target,
                    "context": "Related work in attention mechanisms and applications...",
                    "is_influential": False
                })
    
    def _generate_additional_authors(self) -> None:
        """Generate additional author information and collaborations."""
        
        # Extract unique authors from papers
        self.authors = {}
        
        for paper in self.all_papers:
            for author_name in paper["authors"]:
                if author_name not in self.authors:
                    # Generate author ID
                    author_id = f"author_{hash(author_name) % 100000:05d}"
                    
                    # Generate realistic author metrics
                    base_papers = random.randint(5, 150)
                    base_citations = random.randint(50, 5000)
                    h_index = min(int(np.sqrt(base_citations / 10)), 80)
                    
                    self.authors[author_name] = {
                        "author_id": author_id,
                        "name": author_name,
                        "paper_count": base_papers,
                        "citation_count": base_citations,
                        "h_index": h_index,
                        "affiliations": self._generate_affiliation(author_name),
                        "research_fields": self._infer_research_fields(author_name)
                    }
    
    def _generate_affiliation(self, author_name: str) -> List[str]:
        """Generate realistic institutional affiliations."""
        institutions = [
            "MIT", "Stanford University", "Google Research", "Microsoft Research",
            "Carnegie Mellon University", "UC Berkeley", "Oxford University", 
            "Harvard University", "ETH Zurich", "University of Toronto",
            "Facebook AI Research", "DeepMind", "OpenAI", "NVIDIA Research",
            "Johns Hopkins University", "Mayo Clinic", "Massachusetts General Hospital"
        ]
        
        # Some authors have multiple affiliations
        num_affiliations = random.choices([1, 2], weights=[0.7, 0.3])[0]
        return random.sample(institutions, num_affiliations)
    
    def _infer_research_fields(self, author_name: str) -> List[str]:
        """Infer research fields based on author name patterns."""
        # This is simplified - in reality we'd use paper content
        field_mapping = {
            "machine learning": ["Machine Learning", "Artificial Intelligence"],
            "computer vision": ["Computer Vision", "Machine Learning"], 
            "neuroscience": ["Neuroscience", "Psychology", "Cognitive Science"],
            "medical": ["Medical Informatics", "Healthcare AI", "Biomedical Engineering"],
            "robotics": ["Robotics", "Computer Vision", "Machine Learning"]
        }
        
        # Default fields for most authors
        return random.choice(list(field_mapping.values()))
    
    def _generate_venue_information(self) -> None:
        """Generate comprehensive venue information."""
        
        # Extract unique venues
        self.venues = {}
        
        venue_types = {
            "Journal of Vision": "Journal",
            "IEEE Conference on Computer Vision and Pattern Recognition": "Conference", 
            "arXiv preprint arXiv:2304.02643": "Preprint",
            "Nature Digital Medicine": "Journal",
            "IEEE Transactions on Pattern Analysis and Machine Intelligence": "Journal",
            "IEEE Transactions on Neural Networks and Learning Systems": "Journal",
            "Advances in Neural Information Processing Systems": "Conference",
            "Nature Neuroscience": "Journal",
            "The Lancet Digital Health": "Journal",
            "International Conference on Machine Learning": "Conference",
            "IEEE Robotics and Automation Letters": "Journal",
            "Psychological Review": "Journal",
            "ACM Transactions on Embedded Computing Systems": "Journal"
        }
        
        for paper in self.all_papers:
            venue_name = paper["venue"]
            if venue_name not in self.venues:
                venue_id = f"venue_{hash(venue_name) % 10000:04d}"
                
                self.venues[venue_name] = {
                    "venue_id": venue_id,
                    "name": venue_name,
                    "venue_type": venue_types.get(venue_name, "Journal"),
                    "paper_count": random.randint(100, 2000),
                    "total_citations": random.randint(10000, 500000),
                    "avg_citations_per_paper": random.randint(10, 250)
                }
    
    def generate_complete_dataset(self, name: str = "demo_citation_network") -> DemoDatasetInfo:
        """
        Generate a complete demo dataset with all components.
        
        Args:
            name: Name for the dataset
            
        Returns:
            DemoDatasetInfo with dataset metadata
        """
        self.logger.info(f"Generating complete demo dataset: {name}")
        
        # Create output directory for this dataset
        dataset_dir = self.output_dir / name
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate all components
        papers_file = dataset_dir / "papers.json"
        citations_file = dataset_dir / "citations.json" 
        authors_file = dataset_dir / "authors.json"
        venues_file = dataset_dir / "venues.json"
        metadata_file = dataset_dir / "dataset_info.json"
        
        # Save papers
        with open(papers_file, 'w') as f:
            json.dump(self.all_papers, f, indent=2)
        
        # Save citations
        with open(citations_file, 'w') as f:
            json.dump(self.citations, f, indent=2)
        
        # Save authors
        with open(authors_file, 'w') as f:
            json.dump(list(self.authors.values()), f, indent=2)
        
        # Save venues
        with open(venues_file, 'w') as f:
            json.dump(list(self.venues.values()), f, indent=2)
        
        # Create dataset info
        fields_covered = list(set(paper["field"] for paper in self.all_papers))
        years = [paper["year"] for paper in self.all_papers]
        year_range = (min(years), max(years))
        
        dataset_info = DemoDatasetInfo(
            name=name,
            description="Curated academic citation network for demonstrating platform capabilities",
            total_papers=len(self.all_papers),
            total_citations=len(self.citations),
            total_authors=len(self.authors),
            total_venues=len(self.venues),
            fields_covered=fields_covered,
            year_range=year_range,
            created_at=datetime.now()
        )
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(asdict(dataset_info), f, indent=2, default=str)
        
        self.logger.info(f"Generated demo dataset with {dataset_info.total_papers} papers, "
                        f"{dataset_info.total_citations} citations")
        
        return dataset_info
    
    def generate_minimal_dataset(self, num_papers: int = 5) -> DemoDatasetInfo:
        """
        Generate a minimal dataset for quick testing.
        
        Args:
            num_papers: Number of papers to include
            
        Returns:
            DemoDatasetInfo with dataset metadata
        """
        # Use first N seed papers
        minimal_papers = self.SEED_PAPERS[:num_papers]
        minimal_citations = [c for c in self.citations 
                           if c["source_id"] in [p["paper_id"] for p in minimal_papers]
                           and c["target_id"] in [p["paper_id"] for p in minimal_papers]]
        
        # Create minimal dataset directory
        dataset_name = f"minimal_demo_{num_papers}papers"
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save minimal components
        with open(dataset_dir / "papers.json", 'w') as f:
            json.dump(minimal_papers, f, indent=2)
        
        with open(dataset_dir / "citations.json", 'w') as f:
            json.dump(minimal_citations, f, indent=2)
        
        # Create minimal dataset info
        dataset_info = DemoDatasetInfo(
            name=dataset_name,
            description=f"Minimal demo dataset with {num_papers} papers for quick testing",
            total_papers=len(minimal_papers),
            total_citations=len(minimal_citations),
            total_authors=len(set([author for paper in minimal_papers for author in paper["authors"]])),
            total_venues=len(set([paper["venue"] for paper in minimal_papers])),
            fields_covered=list(set(paper["field"] for paper in minimal_papers)),
            year_range=(min(p["year"] for p in minimal_papers), max(p["year"] for p in minimal_papers)),
            created_at=datetime.now()
        )
        
        # Save metadata
        with open(dataset_dir / "dataset_info.json", 'w') as f:
            json.dump(asdict(dataset_info), f, indent=2, default=str)
        
        return dataset_info


def create_sample_datasets():
    """Create standard sample datasets for the platform."""
    generator = DemoDatasetGenerator()
    
    logger.info("Creating demo datasets...")
    
    # Create complete dataset
    complete_info = generator.generate_complete_dataset("complete_demo")
    logger.info(f"Created complete demo dataset: {complete_info.name}")
    
    # Create minimal dataset for testing
    minimal_info = generator.generate_minimal_dataset(5)
    logger.info(f"Created minimal demo dataset: {minimal_info.name}")
    
    return complete_info, minimal_info


def get_available_datasets(data_dir: Optional[Path] = None) -> List[DemoDatasetInfo]:
    """
    Get list of available demo datasets.
    
    Args:
        data_dir: Directory to search for datasets
        
    Returns:
        List of available dataset info objects
    """
    data_dir = data_dir or Path("data/demo_datasets")
    
    if not data_dir.exists():
        return []
    
    datasets = []
    for dataset_path in data_dir.iterdir():
        if dataset_path.is_dir():
            metadata_file = dataset_path / "dataset_info.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    # Convert back to DemoDatasetInfo
                    metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
                    dataset_info = DemoDatasetInfo(**metadata)
                    datasets.append(dataset_info)
                    
                except Exception as e:
                    logger.warning(f"Failed to load dataset metadata from {metadata_file}: {e}")
    
    return sorted(datasets, key=lambda x: x.created_at, reverse=True)


if __name__ == "__main__":
    # Generate demo datasets when run directly
    create_sample_datasets()