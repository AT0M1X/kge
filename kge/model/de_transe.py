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
        raise NotImplementedError
