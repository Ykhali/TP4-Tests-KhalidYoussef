package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.KhalidYoussef.Assistant;

import javax.print.Doc;
import javax.swing.text.html.parser.DocumentParser;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {
    public static void main(String[] args) throws URISyntaxException {
        System.out.println(" ==== Test 1: Décomposition des différentes tâches ====");

        String Key = System.getenv("GEMINI_KEY");
        if (Key == null || Key.isEmpty()) {
            System.err.println("ERREUR: La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(Key)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        //Phase 1 enregistrement des embeddings
        System.out.println("\n--- Phase 1 : Ingestion ---");
        ApacheTikaDocumentParser documentParser = new ApacheTikaDocumentParser();
        Path pdf = Paths.get("src/main/resources/LangChain4j.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(pdf, documentParser);
        System.out.println("Document chargé: " + document.metadata());

        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300,30);
        List<TextSegment> segments = documentSplitter.split(document);
        System.out.println("Document découpé en " + segments.size() + " segments.");

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        Response<List<dev.langchain4j.data.embedding.Embedding>> response = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        System.out.println("Génération et stockage des embeddings...");
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Phase 1 - Ingestion terminée.");

        //Phase 2: utilisation des embeddings pour répondre aux questions.
        System.out.println("\n--- Phase 2 : Récupération et Conversation ---");
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                // Configuration demandée : 2 résultats les plus pertinents
                .maxResults(2)
                // Configuration demandée : score minimal de 0.5
                .minScore(0.5)
                .build();
        System.out.println("ContentRetriever configuré.");
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatMemory(chatMemory)
                .chatModel(model)
                .contentRetriever(contentRetriever)
                .build();
        System.out.println("Assistant prêt. Vous pouvez commencer à poser des questions.");
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("\n==================================================");
                System.out.println("Posez votre question");
                String question = scanner.nextLine();

                if ("bye".equalsIgnoreCase(question)) break;
                if (question.isBlank()) continue;

                String reponse = assistant.chat(question);

                System.out.println("--------------------------------------------------");
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }

}
