package com.Fintech.OnlineBanking.service;

import java.util.Collection;



import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.Fintech.OnlineBanking.domain.CardType;
import com.Fintech.OnlineBanking.projection.TypeCardProjection;
import com.Fintech.OnlineBanking.projection.AccountTypeProjection;
import com.Fintech.OnlineBanking.repository.cardTypeRepository;

@Service
public class CardTypesService {
	@Autowired
	cardTypeRepository cardTypeRepo;

	public Collection<CardType>  showAllCardTypes() {
		Collection<CardType> allCardType =cardTypeRepo.findAll();
		return allCardType  ;
	}
 

	

}
