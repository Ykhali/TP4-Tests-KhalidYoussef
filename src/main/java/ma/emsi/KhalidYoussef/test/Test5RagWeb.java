package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.KhalidYoussef.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
public class Test5RagWeb {
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.setUseParentHandlers(false);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) {
        configureLogger();
        String key = System.getenv("GEMINI_KEY");
        if (key == null) {
            throw new IllegalStateException("La variable d'environnement GEMINI_KEY n'est pas définie");
        }

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        //Phase 1 enregistrement des embeddings
        System.out.println("\n--- Phase 1 : Ingestion ---");
        ApacheTikaDocumentParser documentParser = new ApacheTikaDocumentParser();
        Path pdf = Paths.get("src/main/resources/FinetuningEtRAG.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(pdf, documentParser);
        System.out.println("Document chargé: " + document.metadata());

        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300,30);
        List<TextSegment> segments = documentSplitter.split(document);
        System.out.println("Document découpé en " + segments.size() + " segments.");

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        System.out.println("Génération et stockage des embeddings...");
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Phase 1 - Ingestion terminée.");

        //Phase 2: ajout d'un ContentRetriever Web (Tavily) + QueryRouter

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                // 2 résultats les plus pertinents
                .maxResults(2)
                // score minimal de 0.5
                .minScore(0.5)
                .build();
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null) {
            throw new IllegalStateException("La variable d'environnement TAVILY_API_KEY n'est pas définie");
        }

        WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(tavilyEngine)
                        .build();

        QueryRouter queryRouter = new DefaultQueryRouter(contentRetriever, webRetriever);

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chatModel)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("\nPoser Votre question : ");
                String question = scanner.nextLine();

                if ("fin".equalsIgnoreCase(question) ||
                        "exit".equalsIgnoreCase(question) ||
                        "bye".equalsIgnoreCase(question) ||
                        "quit".equalsIgnoreCase(question)) {
                    break;
                }
                if (question.isBlank()) {
                    continue;
                }

                String reponse = assistant.chat(question);
                System.out.println("--------------------------------------------------");
                System.out.println("Assistant : " + reponse);
                System.out.println("--------------------------------------------------");
            }
        }
    }
}
