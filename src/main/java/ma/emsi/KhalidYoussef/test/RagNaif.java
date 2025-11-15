package ma.emsi.KhalidYoussef.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import javax.print.Doc;
import javax.swing.text.html.parser.DocumentParser;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

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
    }

}
