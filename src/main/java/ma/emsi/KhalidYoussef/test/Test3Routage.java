package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.KhalidYoussef.Assistant;

import java.nio.file.Path;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test3Routage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);

        System.out.println("--- Logging LangChain4j activé ---");
    }

    private static void ingest(Path path, EmbeddingStore<TextSegment> store, EmbeddingModel embModel) {
        Document document = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(400, 100).split(document);

        if (segments == null || segments.isEmpty()) {
            System.out.println("Aucun segment trouvé pour : " + path);
            return;
        }

        System.out.println("Début de l'ingestion pour : " + path + " (" + segments.size() + " segments)");
        for (TextSegment segment : segments) {
            try {
                Embedding embedding = embModel.embed(segment.text()).content();
                store.add(embedding, segment);

            } catch (Exception e) {
                System.err.println("  Erreur lors de l'embedding du segment "  + e.getMessage());
                System.err.println("  Nouvelle tentative dans 3 secondes...");
                try {
                    Embedding embedding = embModel.embed(segment.text()).content();
                    store.add(embedding, segment);
                } catch (Exception e2) {
                    System.err.println("Échec après nouvelle tentative. Segment ignoré.");
                }
            }
        }

        System.out.println("✅ Ingest terminé pour : " + path);
    }
    static class RouterDecisionCapture {
        private String selectedRetriever;
        private String justification;

        public void setDecision(String retriever, String justification) {
            this.selectedRetriever = retriever;
            this.justification = justification;
        }

        public String getSelectedRetriever() {
            return selectedRetriever;
        }

        public String getJustification() {
            return justification;
        }

        public void reset() {
            selectedRetriever = null;
            justification = null;
        }
    }
    static class CustomQueryRouter implements QueryRouter {
        private final ChatModel chatModel;
        private final Map<ContentRetriever, String> descriptions;
        private final RouterDecisionCapture decisionCapture;

        public CustomQueryRouter(ChatModel chatModel,
                                 Map<ContentRetriever, String> descriptions,
                                 RouterDecisionCapture decisionCapture) {
            this.chatModel = chatModel;
            this.descriptions = descriptions;
            this.decisionCapture = decisionCapture;
        }

        @Override
        public Collection<ContentRetriever> route(dev.langchain4j.rag.query.Query query) {
            StringBuilder promptBuilder = new StringBuilder();
            promptBuilder.append("Tu es un système de routage intelligent. Analyse la question suivante et choisis ");
            promptBuilder.append("le retriever le plus approprié parmi les options disponibles.\n\n");

            int index = 1;
            Map<Integer, ContentRetriever> indexToRetriever = new HashMap<>();

            promptBuilder.append("Options disponibles:\n");
            for (Map.Entry<ContentRetriever, String> entry : descriptions.entrySet()) {
                promptBuilder.append(index).append(") ").append(entry.getValue()).append("\n");
                indexToRetriever.put(index, entry.getKey());
                index++;
            }

            promptBuilder.append("\nQuestion de l'utilisateur: ").append(query.text()).append("\n\n");
            promptBuilder.append("Réponds UNIQUEMENT avec le format suivant:\n");
            promptBuilder.append("RETRIEVER: [numéro du retriever choisi]\n");
            promptBuilder.append("JUSTIFICATION: [explication claire et concise de ton choix en 1-2 phrases]");

            String promptComplet = promptBuilder.toString();
            System.out.println("\n" + "=".repeat(80));
            System.out.println("PROMPT DE ROUTAGE ENVOYÉ AU LLM");
            System.out.println("=".repeat(80));
            System.out.println(promptComplet);
            System.out.println("=".repeat(80) + "\n");

            String llmResponse = "";
            try {
                // Appel simple au ChatModel (compatible avec votre version)
                llmResponse = chatModel.chat(promptComplet);
            } catch (Exception e) {
                System.err.println("Erreur lors de l'appel au LLM pour le routage: " + e.getMessage());
                return List.of(descriptions.keySet().iterator().next());
            }

            String selectedRetriever = "Inconnu";
            String justification = "Aucune justification fournie";
            ContentRetriever chosen = null;

            String[] lines = llmResponse.split("\n");
            for (String line : lines) {
                if (line.startsWith("RETRIEVER:")) {
                    String retrieverNum = line.substring("RETRIEVER:".length()).trim();
                    try {
                        int num = Integer.parseInt(retrieverNum);
                        chosen = indexToRetriever.get(num);
                        selectedRetriever = "Retriever " + num;
                    } catch (NumberFormatException e) {
                        System.err.println("Erreur de parsing du numéro de retriever: " + retrieverNum);
                    }
                } else if (line.startsWith("JUSTIFICATION:")) {
                    justification = line.substring("JUSTIFICATION:".length()).trim();
                }
            }

            decisionCapture.setDecision(selectedRetriever, justification);
            System.out.println("    DÉCISION DU ROUTER    ");
            System.out.println("==========================================");
            System.out.println("Retriever choisi: " + selectedRetriever);
            System.out.println("Justification: " + justification);
            System.out.println("==========================================\n");

            if (chosen != null) {
                return List.of(chosen);
            } else {
                System.err.println("Erreur: Impossible de déterminer le retriever, utilisation du premier par défaut");
                return List.of(descriptions.keySet().iterator().next());
            }
        }
    }

    public static void main(String[] args) {
        configureLogger();

        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("ERREUR: la variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Chat model (Gemini) utilisé pour le routage et pour l'assistant
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .build();

        // Embedding model (Gemini embeddings / Google embeddings)
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(apiKey)
                .modelName("text-embedding-004")
                .build();

        // Mémoire du chat
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // PHASE 1 : ingestion — 2 EmbeddingStores séparés
        EmbeddingStore<TextSegment> storeCours = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeAutre = new InMemoryEmbeddingStore<>();

        System.out.println("\nDébut de l'ingestion des documents...\n");

        ingest(Path.of("src/main/resources/FinetuningEtRAG.pdf"), storeCours, embeddingModel);
        ingest(Path.of("src/main/resources/dynamiqueDesGroupes.pdf"), storeAutre, embeddingModel);


        // PHASE 2 : création des ContentRetrievers (un par store)
        ContentRetriever retrieverCours = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCours)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.0)
                .build();

        ContentRetriever retrieverAutre = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeAutre)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.0)
                .build();

        // Descriptions (Map<ContentRetriever, String>) — l'input lu par le LLM pour décider
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverCours,
                "Documents sur le RAG, fine-tuning et architectures d'IA. Exemples de questions adaptées : "
                        + "'Qu'est-ce que le fine-tuning ?', 'Comment fonctionne Retrieval-Augmented Generation ?', "
                        + "'Quels sont les avantages du RAG ?'");
        descriptions.put(retrieverAutre,
                "Documents sur la dynamique des groupes (théories, influence sociale, 3 pères fondateurs). "
                        + "Exemples : 'Quels sont les stades de formation d'un groupe ?', 'Quelles sont les théories de Tuckman ?'");

        // Capture pour les décisions du router
        RouterDecisionCapture decisionCapture = new RouterDecisionCapture();

        // QueryRouter personnalisé avec capture de justification
        QueryRouter router = new CustomQueryRouter(chatModel, descriptions, decisionCapture);

        // RetrievalAugmentor qui utilisera le QueryRouter pour décider quelles sources interroger
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // Création de l'assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(augmentor)
                .chatMemory(chatMemory)
                .build();

        // Boucle interactive
        System.out.println("\n=============================================================");
        System.out.println("     Assistant RAG avec Routage Intelligent Prêt           ");
        System.out.println("     Tapez 'fin', 'exit', 'quitter' ou 'bye' pour terminer     ");
        System.out.println("===============================================================\n");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.print("\n💬 [USER] > ");
                String question = scanner.nextLine();
                if (question == null) break;
                question = question.trim();
                if (question.isEmpty()) continue;
                if ("fin".equalsIgnoreCase(question) ||
                        "exit".equalsIgnoreCase(question) ||
                        "quitter".equalsIgnoreCase(question) ||
                        "bye".equalsIgnoreCase(question)) {
                    break;
                }

                // Reset de la capture
                decisionCapture.reset();

                // Appel à l'assistant (le router sera appelé automatiquement)
                try {
                    String reponse = assistant.chat(question);
                    System.out.println("\n🤖 [ASSISTANT] : " + reponse);

                    // Affichage récapitulatif de la décision
                    if (decisionCapture.getSelectedRetriever() != null) {
                        System.out.println("\n📊 Récapitulatif: " + decisionCapture.getSelectedRetriever() +
                                " utilisé → " + decisionCapture.getJustification());
                    }
                } catch (Exception e) {
                    System.err.println("  Erreur lors de l'appel à l'assistant : " + e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        System.out.println("\n👋 Fin du TestRoutage. Au revoir!");
        System.exit(0);
    }
}