"""
Data Fixtures for Academic Citation Platform.

This module provides quick-start data fixtures for testing and development.
Fixtures are lightweight, fast-loading datasets designed for:

- Unit testing
- Development testing
- CI/CD pipelines  
- Quick demonstrations
- Integration testing
- Performance benchmarking

All fixtures are self-contained and require no external dependencies.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import random

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FixtureInfo:
    """Information about a test fixture."""
    
    name: str
    description: str
    papers_count: int
    citations_count: int
    authors_count: int
    use_case: str
    load_time_ms: float = 0.0


class DataFixtures:
    """
    Fast-loading data fixtures for testing and development.
    
    Provides minimal but realistic datasets that can be loaded instantly
    for testing various platform components.
    """
    
    @staticmethod
    def minimal_network() -> Dict[str, Any]:
        """
        Minimal citation network with 3 papers and 2 citations.
        Perfect for unit tests and basic functionality verification.
        """
        papers = [
            {
                "paper_id": "paper_001",
                "title": "Foundational Paper in Machine Learning",
                "abstract": "This seminal work establishes the theoretical foundations of modern machine learning algorithms.",
                "authors": ["Dr. Alice Smith", "Prof. Bob Johnson"],
                "venue": "Machine Learning Journal",
                "year": 2018,
                "citation_count": 1250,
                "field": "Machine Learning",
                "doi": "10.1234/ml.2018.001"
            },
            {
                "paper_id": "paper_002", 
                "title": "Advanced Deep Learning Techniques",
                "abstract": "We present novel deep learning architectures that improve upon previous state-of-the-art methods.",
                "authors": ["Dr. Carol Davis", "Dr. Alice Smith"],
                "venue": "Neural Information Processing Systems", 
                "year": 2020,
                "citation_count": 567,
                "field": "Machine Learning",
                "doi": "10.1234/nips.2020.002"
            },
            {
                "paper_id": "paper_003",
                "title": "Real-world Applications of ML in Healthcare",
                "abstract": "This paper demonstrates practical applications of machine learning in medical diagnosis and treatment.",
                "authors": ["Dr. Eve Wilson", "Dr. Frank Miller"],
                "venue": "Journal of Medical AI",
                "year": 2022,
                "citation_count": 89,
                "field": "Medical Informatics", 
                "doi": "10.1234/medai.2022.003"
            }
        ]
        
        citations = [
            {
                "source_id": "paper_002",
                "target_id": "paper_001",
                "context": "Building on the foundational work by Smith et al...",
                "is_influential": True
            },
            {
                "source_id": "paper_003",
                "target_id": "paper_001", 
                "context": "The theoretical framework established in this seminal paper...",
                "is_influential": True
            }
        ]
        
        authors = [
            {
                "author_id": "author_001",
                "name": "Dr. Alice Smith",
                "paper_count": 45,
                "citation_count": 2300,
                "h_index": 22,
                "affiliations": ["MIT", "Google Research"]
            },
            {
                "author_id": "author_002", 
                "name": "Prof. Bob Johnson",
                "paper_count": 78,
                "citation_count": 5600,
                "h_index": 34,
                "affiliations": ["Stanford University"]
            },
            {
                "author_id": "author_003",
                "name": "Dr. Carol Davis", 
                "paper_count": 32,
                "citation_count": 1800,
                "h_index": 18,
                "affiliations": ["Carnegie Mellon University"]
            },
            {
                "author_id": "author_004",
                "name": "Dr. Eve Wilson",
                "paper_count": 23,
                "citation_count": 890,
                "h_index": 14,
                "affiliations": ["Johns Hopkins University", "Mayo Clinic"]
            },
            {
                "author_id": "author_005",
                "name": "Dr. Frank Miller",
                "paper_count": 19,
                "citation_count": 670,
                "h_index": 12,
                "affiliations": ["Harvard Medical School"]
            }
        ]
        
        venues = [
            {
                "venue_id": "venue_001",
                "name": "Machine Learning Journal",
                "venue_type": "Journal",
                "paper_count": 1200,
                "total_citations": 45000,
                "avg_citations_per_paper": 37.5
            },
            {
                "venue_id": "venue_002",
                "name": "Neural Information Processing Systems",
                "venue_type": "Conference", 
                "paper_count": 2500,
                "total_citations": 125000,
                "avg_citations_per_paper": 50.0
            },
            {
                "venue_id": "venue_003",
                "name": "Journal of Medical AI",
                "venue_type": "Journal",
                "paper_count": 450,
                "total_citations": 12000,
                "avg_citations_per_paper": 26.7
            }
        ]
        
        return {
            "papers": papers,
            "citations": citations,
            "authors": authors,
            "venues": venues,
            "metadata": {
                "name": "minimal_network",
                "description": "Minimal 3-paper citation network for testing",
                "created_at": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def collaboration_network() -> Dict[str, Any]:
        """
        Network focused on author collaborations across 5 papers.
        Ideal for testing author network analysis features.
        """
        papers = [
            {
                "paper_id": "collab_001",
                "title": "Multi-institutional Study on AI Ethics",
                "abstract": "A comprehensive study involving researchers from five institutions examining ethical implications of AI deployment.",
                "authors": ["Dr. Anna Lee", "Prof. David Chen", "Dr. Maria Rodriguez"],
                "venue": "AI Ethics Quarterly",
                "year": 2021,
                "citation_count": 234,
                "field": "AI Ethics",
                "doi": "10.1234/ethics.2021.001"
            },
            {
                "paper_id": "collab_002",
                "title": "Cross-disciplinary Approaches to Machine Learning",
                "abstract": "This work brings together computer scientists, psychologists, and philosophers to examine ML from multiple perspectives.",
                "authors": ["Prof. David Chen", "Dr. Sarah Kim", "Dr. Michael Brown"],
                "venue": "Interdisciplinary Science Journal", 
                "year": 2021,
                "citation_count": 178,
                "field": "Interdisciplinary Studies",
                "doi": "10.1234/interdisciplinary.2021.002"
            },
            {
                "paper_id": "collab_003",
                "title": "Global Collaboration in AI Research: A Network Analysis",
                "abstract": "We analyze patterns of international collaboration in AI research using network analysis techniques.",
                "authors": ["Dr. Maria Rodriguez", "Dr. James Wilson", "Dr. Anna Lee"],
                "venue": "Science Policy Review",
                "year": 2022,
                "citation_count": 145,
                "field": "Science Policy",
                "doi": "10.1234/policy.2022.003"
            },
            {
                "paper_id": "collab_004", 
                "title": "Psychological Foundations of Human-AI Interaction",
                "abstract": "This paper explores psychological principles underlying effective human-AI collaboration.",
                "authors": ["Dr. Sarah Kim", "Dr. Lisa Johnson", "Prof. David Chen"],
                "venue": "Journal of Human-Computer Studies",
                "year": 2022,
                "citation_count": 112,
                "field": "Human-Computer Interaction", 
                "doi": "10.1234/hci.2022.004"
            },
            {
                "paper_id": "collab_005",
                "title": "The Future of Collaborative AI Research",
                "abstract": "A forward-looking analysis of trends in collaborative AI research based on bibliometric analysis.",
                "authors": ["Dr. Anna Lee", "Dr. Michael Brown", "Dr. James Wilson", "Dr. Lisa Johnson"],
                "venue": "Nature Machine Intelligence",
                "year": 2023, 
                "citation_count": 67,
                "field": "AI Research",
                "doi": "10.1038/natmi.2023.005"
            }
        ]
        
        # More complex citation network
        citations = [
            {"source_id": "collab_002", "target_id": "collab_001", "context": "Building on previous work...", "is_influential": True},
            {"source_id": "collab_003", "target_id": "collab_001", "context": "The ethical framework established...", "is_influential": True},
            {"source_id": "collab_003", "target_id": "collab_002", "context": "Cross-disciplinary approaches...", "is_influential": False},
            {"source_id": "collab_004", "target_id": "collab_002", "context": "The interdisciplinary methods...", "is_influential": True},
            {"source_id": "collab_005", "target_id": "collab_003", "context": "Network analysis reveals...", "is_influential": True},
            {"source_id": "collab_005", "target_id": "collab_004", "context": "Understanding psychological factors...", "is_influential": False},
        ]
        
        authors = [
            {
                "author_id": "collab_auth_001",
                "name": "Dr. Anna Lee", 
                "paper_count": 67,
                "citation_count": 3400,
                "h_index": 28,
                "affiliations": ["MIT", "Berkeley"]
            },
            {
                "author_id": "collab_auth_002",
                "name": "Prof. David Chen",
                "paper_count": 89,
                "citation_count": 4500, 
                "h_index": 32,
                "affiliations": ["Stanford University"]
            },
            {
                "author_id": "collab_auth_003",
                "name": "Dr. Maria Rodriguez",
                "paper_count": 45,
                "citation_count": 2100,
                "h_index": 23,
                "affiliations": ["University of Toronto"]
            },
            {
                "author_id": "collab_auth_004",
                "name": "Dr. Sarah Kim",
                "paper_count": 34,
                "citation_count": 1800,
                "h_index": 19,
                "affiliations": ["Carnegie Mellon University"]
            },
            {
                "author_id": "collab_auth_005", 
                "name": "Dr. Michael Brown",
                "paper_count": 56,
                "citation_count": 2800,
                "h_index": 25,
                "affiliations": ["Oxford University", "DeepMind"]
            },
            {
                "author_id": "collab_auth_006",
                "name": "Dr. James Wilson",
                "paper_count": 41,
                "citation_count": 1900,
                "h_index": 21,
                "affiliations": ["University of Washington"]
            },
            {
                "author_id": "collab_auth_007",
                "name": "Dr. Lisa Johnson",
                "paper_count": 38,
                "citation_count": 1650,
                "h_index": 18,
                "affiliations": ["University of Michigan"]
            }
        ]
        
        return {
            "papers": papers,
            "citations": citations, 
            "authors": authors,
            "venues": [
                {"venue_id": "collab_venue_001", "name": "AI Ethics Quarterly", "venue_type": "Journal"},
                {"venue_id": "collab_venue_002", "name": "Interdisciplinary Science Journal", "venue_type": "Journal"},
                {"venue_id": "collab_venue_003", "name": "Science Policy Review", "venue_type": "Journal"},
                {"venue_id": "collab_venue_004", "name": "Journal of Human-Computer Studies", "venue_type": "Journal"},
                {"venue_id": "collab_venue_005", "name": "Nature Machine Intelligence", "venue_type": "Journal"}
            ],
            "metadata": {
                "name": "collaboration_network",
                "description": "Author collaboration network across 5 papers",
                "created_at": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def temporal_evolution() -> Dict[str, Any]:
        """
        Papers showing temporal evolution of a research field (2015-2024).
        Perfect for testing temporal analysis and trend detection.
        """
        papers = [
            {
                "paper_id": "temporal_001",
                "title": "Early Convolutional Neural Networks: Promise and Limitations", 
                "abstract": "This early work explores the potential of convolutional architectures while acknowledging computational limitations.",
                "authors": ["Dr. Emma Stone", "Prof. Jack Wilson"],
                "venue": "Computer Vision Conference",
                "year": 2015,
                "citation_count": 1245,
                "field": "Computer Vision",
                "doi": "10.1234/cv.2015.001"
            },
            {
                "paper_id": "temporal_002",
                "title": "Breakthrough in Deep Learning: ResNet Architecture",
                "abstract": "We introduce residual connections that enable training of much deeper networks than previously possible.",
                "authors": ["Dr. Ryan Clark", "Dr. Emma Stone", "Dr. Sophie Turner"],
                "venue": "International Conference on Machine Learning",
                "year": 2017,
                "citation_count": 8934,
                "field": "Machine Learning",
                "doi": "10.1234/icml.2017.002"
            },
            {
                "paper_id": "temporal_003",
                "title": "Attention Mechanisms Transform Computer Vision",
                "abstract": "This work demonstrates how attention mechanisms from NLP can revolutionize computer vision tasks.",
                "authors": ["Dr. Alex Kim", "Dr. Maya Patel"],
                "venue": "Neural Information Processing Systems", 
                "year": 2019,
                "citation_count": 3456,
                "field": "Computer Vision",
                "doi": "10.1234/nips.2019.003"
            },
            {
                "paper_id": "temporal_004",
                "title": "Vision Transformers: An Image is Worth 16x16 Words",
                "abstract": "We show that pure transformer architectures can achieve state-of-the-art results on image classification without convolutions.",
                "authors": ["Dr. Sophie Turner", "Dr. Leo Rodriguez", "Dr. Alex Kim"],
                "venue": "International Conference on Learning Representations",
                "year": 2021,
                "citation_count": 5678,
                "field": "Computer Vision",
                "doi": "10.1234/iclr.2021.004"
            },
            {
                "paper_id": "temporal_005",
                "title": "Self-supervised Learning: The Next Frontier", 
                "abstract": "This comprehensive survey explores self-supervised learning methods and their impact across vision and NLP.",
                "authors": ["Dr. Maya Patel", "Dr. Chris Anderson", "Dr. Emma Stone"],
                "venue": "Nature Machine Intelligence",
                "year": 2022,
                "citation_count": 1234,
                "field": "Machine Learning",
                "doi": "10.1038/natmi.2022.005"
            },
            {
                "paper_id": "temporal_006",
                "title": "Foundation Models: Emergent Abilities at Scale",
                "abstract": "We examine the emergent capabilities of large-scale foundation models and their implications for AI.", 
                "authors": ["Dr. Leo Rodriguez", "Dr. Chris Anderson"],
                "venue": "Science",
                "year": 2023,
                "citation_count": 567,
                "field": "Artificial Intelligence",
                "doi": "10.1126/science.2023.006"
            },
            {
                "paper_id": "temporal_007",
                "title": "Multimodal AI: Integrating Vision, Language, and Beyond",
                "abstract": "This work presents unified architectures for processing multiple modalities simultaneously with unprecedented performance.",
                "authors": ["Dr. Ryan Clark", "Dr. Maya Patel", "Dr. Sophie Turner"],
                "venue": "Nature",
                "year": 2024,
                "citation_count": 123,
                "field": "Multimodal AI",
                "doi": "10.1038/nature.2024.007"
            }
        ]
        
        # Complex citation network showing research evolution
        citations = [
            {"source_id": "temporal_002", "target_id": "temporal_001", "context": "Building on early CNN work...", "is_influential": True},
            {"source_id": "temporal_003", "target_id": "temporal_002", "context": "ResNet architectures provide...", "is_influential": True},
            {"source_id": "temporal_004", "target_id": "temporal_003", "context": "Attention mechanisms in vision...", "is_influential": True},
            {"source_id": "temporal_005", "target_id": "temporal_002", "context": "Deep learning foundations...", "is_influential": False},
            {"source_id": "temporal_005", "target_id": "temporal_003", "context": "Attention-based methods...", "is_influential": True},
            {"source_id": "temporal_005", "target_id": "temporal_004", "context": "Vision transformers demonstrate...", "is_influential": True},
            {"source_id": "temporal_006", "target_id": "temporal_005", "context": "Self-supervised learning enables...", "is_influential": True},
            {"source_id": "temporal_007", "target_id": "temporal_004", "context": "Transformer architectures...", "is_influential": True},
            {"source_id": "temporal_007", "target_id": "temporal_006", "context": "Foundation model capabilities...", "is_influential": True}
        ]
        
        return {
            "papers": papers,
            "citations": citations,
            "authors": [
                {"author_id": "temp_auth_001", "name": "Dr. Emma Stone", "paper_count": 78, "citation_count": 4500, "h_index": 32},
                {"author_id": "temp_auth_002", "name": "Prof. Jack Wilson", "paper_count": 156, "citation_count": 12000, "h_index": 45},
                {"author_id": "temp_auth_003", "name": "Dr. Ryan Clark", "paper_count": 89, "citation_count": 8900, "h_index": 38},
                {"author_id": "temp_auth_004", "name": "Dr. Sophie Turner", "paper_count": 67, "citation_count": 5600, "h_index": 34},
                {"author_id": "temp_auth_005", "name": "Dr. Alex Kim", "paper_count": 45, "citation_count": 3400, "h_index": 28},
                {"author_id": "temp_auth_006", "name": "Dr. Maya Patel", "paper_count": 54, "citation_count": 4200, "h_index": 30},
                {"author_id": "temp_auth_007", "name": "Dr. Leo Rodriguez", "paper_count": 43, "citation_count": 2800, "h_index": 25},
                {"author_id": "temp_auth_008", "name": "Dr. Chris Anderson", "paper_count": 38, "citation_count": 2100, "h_index": 22}
            ],
            "venues": [
                {"venue_id": "temp_venue_001", "name": "Computer Vision Conference", "venue_type": "Conference"},
                {"venue_id": "temp_venue_002", "name": "International Conference on Machine Learning", "venue_type": "Conference"},
                {"venue_id": "temp_venue_003", "name": "Neural Information Processing Systems", "venue_type": "Conference"},
                {"venue_id": "temp_venue_004", "name": "International Conference on Learning Representations", "venue_type": "Conference"},
                {"venue_id": "temp_venue_005", "name": "Nature Machine Intelligence", "venue_type": "Journal"},
                {"venue_id": "temp_venue_006", "name": "Science", "venue_type": "Journal"},
                {"venue_id": "temp_venue_007", "name": "Nature", "venue_type": "Journal"}
            ],
            "metadata": {
                "name": "temporal_evolution",
                "description": "Temporal evolution of computer vision research (2015-2024)",
                "created_at": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def performance_benchmark() -> Dict[str, Any]:
        """
        Large fixture for performance testing with 50+ papers.
        Used for benchmarking import, search, and analysis performance.
        """
        papers = []
        citations = []
        authors = []
        venues = []
        
        # Generate 50 papers across different fields and years
        fields = ["Machine Learning", "Computer Vision", "Natural Language Processing", 
                 "Robotics", "AI Ethics", "Medical Informatics"]
        venue_names = ["ICML", "NeurIPS", "ICLR", "CVPR", "EMNLP", "Nature", "Science"]
        author_pool = [f"Dr. {name}" for name in [
            "Alice Johnson", "Bob Smith", "Carol Davis", "David Lee", "Eva Martinez",
            "Frank Wilson", "Grace Chen", "Henry Brown", "Iris Kim", "Jack Taylor",
            "Kate Anderson", "Liam Rodriguez", "Maya Patel", "Noah Thompson", "Olivia White"
        ]]
        
        for i in range(50):
            paper_id = f"perf_{i+1:03d}"
            field = random.choice(fields)
            year = random.randint(2015, 2024)
            citation_count = random.randint(10, 2000)
            
            # Select 2-4 random authors
            paper_authors = random.sample(author_pool, random.randint(2, 4))
            
            papers.append({
                "paper_id": paper_id,
                "title": f"Research Paper {i+1}: {field} Innovations",
                "abstract": f"This paper presents novel {field.lower()} techniques with significant improvements over existing methods. Our approach demonstrates {random.randint(5, 25)}% improvement on standard benchmarks.",
                "authors": paper_authors,
                "venue": random.choice(venue_names),
                "year": year,
                "citation_count": citation_count,
                "field": field,
                "doi": f"10.1234/{field.lower().replace(' ', '')}.{year}.{i+1:03d}"
            })
        
        # Generate random citation network
        for i in range(80):  # 80 citations
            source = random.choice(papers)
            target = random.choice(papers)
            
            if source["paper_id"] != target["paper_id"]:  # No self-citations
                citations.append({
                    "source_id": source["paper_id"],
                    "target_id": target["paper_id"],
                    "context": f"This work builds on {target['title'][:30]}...",
                    "is_influential": random.choice([True, False])
                })
        
        # Generate authors with metrics
        for author_name in author_pool:
            author_id = f"perf_auth_{hash(author_name) % 1000:03d}"
            authors.append({
                "author_id": author_id,
                "name": author_name,
                "paper_count": random.randint(20, 100),
                "citation_count": random.randint(500, 8000),
                "h_index": random.randint(15, 45),
                "affiliations": random.sample(["MIT", "Stanford", "Google", "Microsoft", "CMU"], 
                                            random.randint(1, 2))
            })
        
        # Generate venues
        for venue_name in venue_names:
            venues.append({
                "venue_id": f"perf_venue_{hash(venue_name) % 100:02d}",
                "name": venue_name,
                "venue_type": "Conference" if venue_name in ["ICML", "NeurIPS", "ICLR", "CVPR", "EMNLP"] else "Journal",
                "paper_count": random.randint(500, 3000),
                "total_citations": random.randint(50000, 500000),
                "avg_citations_per_paper": random.randint(50, 200)
            })
        
        return {
            "papers": papers,
            "citations": citations,
            "authors": authors,
            "venues": venues,
            "metadata": {
                "name": "performance_benchmark",
                "description": "Large dataset with 50+ papers for performance testing",
                "created_at": datetime.now().isoformat()
            }
        }


class FixtureManager:
    """
    Manager for creating and loading test fixtures.
    
    Provides convenient access to all fixtures with caching and
    temporary file management.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._temp_dir: Optional[Path] = None
    
    def get_fixture(self, name: str) -> Dict[str, Any]:
        """
        Get fixture data by name.
        
        Args:
            name: Fixture name (minimal_network, collaboration_network, 
                  temporal_evolution, performance_benchmark)
                  
        Returns:
            Fixture data dictionary
        """
        if name in self._cache:
            return self._cache[name]
        
        fixture_methods = {
            "minimal_network": DataFixtures.minimal_network,
            "collaboration_network": DataFixtures.collaboration_network, 
            "temporal_evolution": DataFixtures.temporal_evolution,
            "performance_benchmark": DataFixtures.performance_benchmark
        }
        
        if name not in fixture_methods:
            raise ValueError(f"Unknown fixture: {name}. Available: {list(fixture_methods.keys())}")
        
        start_time = datetime.now()
        fixture_data = fixture_methods[name]()
        load_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Add load time to metadata
        fixture_data["metadata"]["load_time_ms"] = load_time
        
        # Cache the result
        self._cache[name] = fixture_data
        
        self.logger.info(f"Loaded fixture '{name}' in {load_time:.2f}ms")
        return fixture_data
    
    def save_fixture_to_file(self, name: str, output_path: Optional[Path] = None) -> Path:
        """
        Save fixture to JSON file.
        
        Args:
            name: Fixture name
            output_path: Output file path (generates temp file if None)
            
        Returns:
            Path to saved file
        """
        fixture_data = self.get_fixture(name)
        
        if output_path is None:
            if self._temp_dir is None:
                self._temp_dir = Path(tempfile.mkdtemp(prefix="fixtures_"))
            output_path = self._temp_dir / f"{name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(fixture_data, f, indent=2)
        
        self.logger.info(f"Saved fixture '{name}' to {output_path}")
        return output_path
    
    def get_fixture_info(self, name: str) -> FixtureInfo:
        """Get information about a fixture."""
        fixture_data = self.get_fixture(name)
        metadata = fixture_data["metadata"]
        
        return FixtureInfo(
            name=name,
            description=metadata["description"],
            papers_count=len(fixture_data["papers"]),
            citations_count=len(fixture_data["citations"]),
            authors_count=len(fixture_data["authors"]),
            use_case=self._infer_use_case(name),
            load_time_ms=metadata.get("load_time_ms", 0.0)
        )
    
    def _infer_use_case(self, name: str) -> str:
        """Infer use case from fixture name."""
        use_cases = {
            "minimal_network": "Unit testing, basic functionality",
            "collaboration_network": "Author network analysis, collaboration patterns",
            "temporal_evolution": "Temporal analysis, trend detection", 
            "performance_benchmark": "Performance testing, scalability"
        }
        return use_cases.get(name, "General testing")
    
    def list_available_fixtures(self) -> List[FixtureInfo]:
        """Get list of all available fixtures."""
        fixture_names = ["minimal_network", "collaboration_network", 
                        "temporal_evolution", "performance_benchmark"]
        
        return [self.get_fixture_info(name) for name in fixture_names]
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self.logger.info("Cleaned up temporary fixture files")


# Global fixture manager
_fixture_manager: Optional[FixtureManager] = None


def get_fixture_manager() -> FixtureManager:
    """Get global fixture manager instance."""
    global _fixture_manager
    
    if _fixture_manager is None:
        _fixture_manager = FixtureManager()
    
    return _fixture_manager


def quick_fixture(name: str) -> Dict[str, Any]:
    """
    Quick function to get fixture data.
    
    Args:
        name: Fixture name
        
    Returns:
        Fixture data dictionary
    """
    return get_fixture_manager().get_fixture(name)


if __name__ == "__main__":
    # Demonstrate fixture usage
    manager = get_fixture_manager()
    
    print("Available Fixtures:")
    for info in manager.list_available_fixtures():
        print(f"  {info.name}: {info.description}")
        print(f"    Papers: {info.papers_count}, Citations: {info.citations_count}")
        print(f"    Use case: {info.use_case}")
        print()
    
    # Load and display minimal network
    minimal = manager.get_fixture("minimal_network")
    print(f"Minimal network loaded: {len(minimal['papers'])} papers, {len(minimal['citations'])} citations")