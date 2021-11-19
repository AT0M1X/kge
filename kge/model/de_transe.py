import torch
from torch.tensor import Tensor
from kge.model.transe import TransE


class DETransE(TransE):
    def score_spo(
        self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, direction=None
    ) -> Tensor:
        s = self.get_s_embedder().embed(s, t)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o, t)
        return self._scorer.score_emb(s, p, o, combine="spo").view(-1)

    def score_sp(self, s: Tensor, p: Tensor, t: Tensor, o: Tensor = None) -> Tensor:
        s = self.get_s_embedder().embed(s, t)
        p = self.get_p_embedder().embed(p)
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o, t)

        return self._scorer.score_emb(s, p, o, combine="sp_")

    def score_po(self, p: Tensor, o: Tensor, t: Tensor, s: Tensor = None) -> Tensor:
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s, t)
        o = self.get_o_embedder().embed(o, t)
        p = self.get_p_embedder().embed(p)

    def score_so(self, s: Tensor, o: Tensor, t: Tensor, p: Tensor = None) -> Tensor:
        s = self.get_s_embedder().embed(s, t)
        o = self.get_o_embedder().embed(o, t)
        if p is None:
            p = self.get_p_embedder().embed_all()
        else:
            p = self.get_p_embedder().embed(p)

        return self._scorer.score_emb(s, p, o, combine="s_o")

    def score_sp_po(
        self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, entity_subset: Tensor = None
    ) -> Tensor:
        s = self.get_s_embedder().embed(s, t)
        p = self.get_p_embedder().embed(p)
        o = self.get_o_embedder().embed(o, t)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset, t)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
            po_scores = self._scorer.score_emb(all_entities, p, o, combine="_po")
        else:
            if entity_subset is not None:
                all_objects = self.get_o_embedder().embed(entity_subset, t)
                all_subjects = self.get_s_embedder().embed(entity_subset, t)
            else:
                all_objects = self.get_o_embedder().embed_all()
                all_subjects = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp_")
            po_scores = self._scorer.score_emb(all_subjects, p, o, combine="_po")
        return torch.cat((sp_scores, po_scores), dim=1)
